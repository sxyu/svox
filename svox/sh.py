"""
[BSD 2-CLAUSE LICENSE]

Copyright Alex Yu 2021

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from math import sqrt, pi
import torch

C0 = 0.5 * sqrt(1.0 / pi)
C1 = sqrt(3 / (4 * pi))
C2 = [0.5 * sqrt(15.0 / pi),
        -0.5 * sqrt(15.0 / pi),
        0.25 * sqrt(5.0 / pi),
        -0.5 * sqrt(15.0 / pi),
        0.25 * sqrt(15.0 / pi)]

C3 = [-0.25 * sqrt(35 / (2 * pi)),
        0.5 * sqrt(105 / pi),
        -0.25 * sqrt(21/(2 * pi)),
        0.25 * sqrt(7 / pi),
        -0.25 * sqrt(21/(2 * pi)),
        0.25 * sqrt(105 / pi),
        -0.25 * sqrt(35/(2 * pi))
        ]

def eval_sh(order, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.

    :param order: int SH order. Currently, 0-3 supported
    :param sh: torch.Tensor SH coeffs (..., C, (order + 1) ** 2)
    :param dirs: torch.Tensor unit directions (..., 3)

    :return: (..., C)
    """
    assert order <= 3 and order >= 0
    assert (order + 1) ** 2 == sh.shape[-1]
    C = sh.shape[-2]

    result = C0 * sh[..., 0]
    if order > 0:
        x = dirs[..., 0:1]
        y = dirs[..., 1:2]
        z = dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
        if order > 1:
            xx, yy, zz = x * x, y * y, z * z
            result = (result +
                    C2[0] * x * y * sh[..., 4] +
                    C2[1] * y * z * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * x * z * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if order > 2:
                tmp_zzxxyy = 4 * zz - xx - yy
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * x * y * z * sh[..., 10] +
                        C3[2] * y * tmp_zzxxyy * sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * tmp_zzxxyy * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])

    return result


def eval_sh_bases(order, dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be 
    obtained through simple multiplication.

    :param order: int SH order. Currently, 0-3 supported
    :param dirs: jnp.ndarray (..., 3) unit directions

    :return: jnp.ndarray (..., (order+1) ** 2)
    """
    assert order <= 3 and order >= 0
    result = [torch.full(dirs.shape[:-1], C0, device=dirs.device)]
    if order > 0:
        x = dirs[..., 0]
        y = dirs[..., 1]
        z = dirs[..., 2]
        result.extend([-C1 * y, C1 * z, - C1 * x])
        if order > 1:
            xx, yy, zz = x * x, y * y, z * z
            result.extend([
                    C2[0] * x * y,
                    C2[1] * y * z,
                    C2[2] * (2.0 * zz - xx - yy),
                    C2[3] * x * z,
                    C2[4] * (xx - yy)])

            if order > 2:
                tmp_zzxxyy = 4 * zz - xx - yy
                result.extend([
                        C3[0] * y * (3 * xx - yy),
                        C3[1] * x * y * z,
                        C3[2] * y * tmp_zzxxyy,
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                        C3[4] * x * tmp_zzxxyy,
                        C3[5] * z * (xx - yy),
                        C3[6] * x * (xx - 3 * yy)])

    return torch.stack(result, dim=-1)
