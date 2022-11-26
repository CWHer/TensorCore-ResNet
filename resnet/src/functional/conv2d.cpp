/** @file conv2d.cpp
*/

#include "common.h"
#include "tensor.hpp"
#include "functional/conv2d.hpp"

/**
 * Applies a 2D convolution over an input image composed of several input planes.
 *
 * See :class:`~Conv2d` for details and output shape.
 *
 * @param input input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
 * @param weight filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
 * @param bias optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
 * @param stride the stride of the convolving kernel. Can be a single number or a tuple `(sH, sW)`. Default: 1
 * @param padding implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a tuple `(padH, padW)`. Default: 0

      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.

 * @param dilation the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1
 * @param groups split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
 * @return
 */
Tensor conv2d(const Tensor &input,
              const Tensor &weight,
              const Tensor &bias,
              int64_t stride,
              int64_t padding,
              int64_t dilation,
              int64_t groups) {

}
