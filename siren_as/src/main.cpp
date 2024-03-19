#include <iostream>
#include <cmath>
#include <stdexcept>

#include "Application.h"
#include "nsdf.h"

int main (int argc, char *argv[])
{
  if (argc != 4) {
    std::cout << "Example usage: ./siren_bvh res/sdf1_arch.txt res/sdf1_test.bin res/sdf1_weights.bin" << std::endl;
    return 1;
  }

  nsdf::SDFArch sdf_arch;
  sdf_arch.from_file(argv[1]);

  nsdf::SDFTest sdf_test;
  sdf_test.from_file(argv[2]);

  nsdf::SIREN siren_nn;
  siren_nn.from_file(argv[3], sdf_arch);

  try
  {
    nsdf::Application app{};
    float total_error = 0.0f;

    for (size_t i = 0; i < sdf_test.points_count; i++) {
      std::vector<float> point {
        sdf_test.points[3 * i + 0],
        sdf_test.points[3 * i + 1],
        sdf_test.points[3 * i + 2]
      };
      float dist = siren_nn.forward(point);
      total_error += std::abs(dist - sdf_test.etalon_dists[i]);
    }

    std::cout << "Error: " << total_error / sdf_test.points_count << std::endl;

    // 0. Adjust regular grid scale
    // 1. Prepare regular grid points array
    // 2. Calculate distances for each point
    // 2. Create bounding boxes for 
  }
  catch (const std::runtime_error& e)
  {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

