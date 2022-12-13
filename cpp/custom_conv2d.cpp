#include <torch/extension.h>

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
  auto output = torch::zeros({n_items, output_h, output_w, output_c});

  // Switch to h, w, c layout.
  const auto input_hwc = input.permute({0, 2, 3, 1}).contiguous();

  // Switch to c_out, h, w, c_in layout.
  const auto kernel_hwc = kernel.permute({0, 2, 3, 1}).contiguous();

  // Use accessors for more efficient element-wise operations.
  const auto input_accessor = input_hwc.accessor<float, 4>();
  const auto kernel_accessor = kernel_hwc.accessor<float, 4>();
  auto output_accessor = output.accessor<float, 4>();

  // Iterate over output locations.
  for (int item = 0; item < n_items; item++) {
    for (int y_out = 0; y_out < output_h; y_out++) {
      const int y_start = y_out * stride_y;
      for (int x_out = 0; x_out < output_w; x_out++) {
        const int x_start = x_out * stride_x;
        for (int c_out = 0; c_out < output_c; c_out++) {

          // Iterate over elements of the kernel.
          float sum = 0.0;
          for (int kernel_y = 0; kernel_y < kernel_h; kernel_y++) {
            const int y_in = y_start + kernel_y;
            for (int kernel_x = 0; kernel_x < kernel_w; kernel_x++) {
              const int x_in = x_start + kernel_x;
              const float *const input_ptr =
                  &(input_accessor[item][y_in][x_in][0]);
              const float *const kernel_ptr =
                  &(kernel_accessor[c_out][kernel_y][kernel_x][0]);
              for (int c_in = 0; c_in < input_c; c_in++) {
                sum += input_ptr[c_in] * kernel_ptr[c_in];
              }
            }
          }
          output_accessor[item][y_out][x_out][c_out] = sum;
        }
      }
    }
  }

  // Switch back to c, h, w, layout (what PyTorch expects).
  return output.permute({0, 3, 1, 2});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward, "custom_conv2d forward");
}
