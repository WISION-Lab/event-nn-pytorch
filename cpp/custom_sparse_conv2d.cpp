#include <torch/extension.h>

using namespace torch::indexing;

torch::Tensor forward(const torch::Tensor input, const torch::Tensor kernel,
                      const std::vector<int> stride) {

  // Compute the geometry of the convolution.
  const int n_items = input.size(0);
  const int input_c = input.size(1);
  const int input_h = input.size(2);
  const int input_w = input.size(3);
  const int kernel_h = kernel.size(2);
  const int kernel_w = kernel.size(3);
  const int stride_y = stride[0];
  const int stride_x = stride[1];
  const int output_h = (input_h - kernel_h) / stride_y + 1;
  const int output_w = (input_w - kernel_w) / stride_x + 1;
  const int output_c = kernel.size(0);

  // Allocate the output tensor with h, w, c layout.
  const int pad_h = kernel_h / stride_y;
  const int pad_w = kernel_w / stride_x;
  auto output = torch::zeros(
      {n_items, output_h + 2 * pad_h, output_w + 2 * pad_w, output_c});

  // Switch to h, w, c layout.
  const auto input_hwc = input.permute({0, 2, 3, 1}).contiguous();

  // Switch to c_in, h, w, c_out layout.
  const auto kernel_hwc = kernel.permute({1, 2, 3, 0}).contiguous();

  // Use accessors for more efficient element-wise operations.
  const auto input_accessor = input_hwc.accessor<float, 4>();
  const auto kernel_accessor = kernel_hwc.accessor<float, 4>();
  auto output_accessor = output.accessor<float, 4>();

  // Iterate over input locations.
  for (int item = 0; item < n_items; item++) {
    for (int y_in = 0; y_in < input_h; y_in++) {
      const int offset_y = y_in % stride_y;
      const int start_y = (y_in - offset_y + kernel_h) / stride_y;
      for (int x_in = 0; x_in < input_w; x_in++) {
        const int offset_x = x_in % stride_x;
        const int start_x = (x_in - offset_x + kernel_w) / stride_x;
        for (int c_in = 0; c_in < input_c; c_in++) {
          const float value = input_accessor[item][y_in][x_in][c_in];

          // Skip zero input entries.
          if (value != 0.0) {

            // Iterate over elements of the kernel.
            for (int kernel_y = offset_y, y_out = start_y; kernel_y < kernel_h;
                 kernel_y += stride_y, y_out--) {
              for (int kernel_x = offset_x, x_out = start_x;
                   kernel_x < kernel_w; kernel_x += stride_x, x_out--) {
                const float *const kernel_ptr =
                    &(kernel_accessor[c_in][kernel_y][kernel_x][0]);
                float *const output_ptr =
                    &(output_accessor[item][y_out][x_out][0]);
                for (int c_out = 0; c_out < output_c; c_out++) {
                  output_ptr[c_out] += value * kernel_ptr[c_out];
                }
              }
            }
          }
        }
      }
    }
  }

  // Switch back to c, h, w, layout (what PyTorch expects) and remove padding.
  output = output.permute({0, 3, 1, 2});
  return output.index({"...", Slice(pad_h, -pad_h), Slice(pad_w, -pad_w)});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward, "custom_sparse_conv2d forward");
}
