#include <iostream>
#include <cstdint>
#include <cmath>
#include <stdexcept>

#include "LBVH.h"
#include "engine/core/GPUContext.h"
#include "engine/util/Paths.h"

#include "nsdf.h"

struct Point3D {
    float x, y, z;
};

struct Element {
    uint32_t primitiveIdx;
    float aabbMinX;
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

std::vector<Point3D> generateRegularGrid(float maxAbsValue, float spacing) {
    std::vector<Point3D> gridPoints;
    int gridSize = static_cast<int>(2 * maxAbsValue / spacing) + 1;
    float offset = maxAbsValue;
    
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            for (int k = 0; k < gridSize; ++k) {
                Point3D point;
                point.x = (i * spacing) - offset;
                point.y = (j * spacing) - offset;
                point.z = (k * spacing) - offset;
                gridPoints.push_back(point);
            }
        }
    }
    
    return gridPoints;
}

std::vector<Point3D> trimPointsWithPositiveSDF(const std::vector<Point3D>& orig, const nsdf::SIREN& siren_nn) {
  std::vector<Point3D> result;
  for (const auto& point : orig) {
    std::vector<float> input {
      point.x, point.y, point.z
    };
    float dist = siren_nn.forward(input);
    if (dist <= 0.0f) {
      result.push_back(point);
    }
  }
  return result;
}

std::vector<Element> generateAABBs(const std::vector<Point3D>& gridPoints, float spacing, float eps) {
    std::vector<Element> aabbs;
    
    for (const auto& point : gridPoints) {
        Element elem;
        elem.primitiveIdx = aabbs.size();
        elem.aabbMinX = point.x - (spacing / 2.0f) - eps;
        elem.aabbMinY = point.y - (spacing / 2.0f) - eps;
        elem.aabbMinZ = point.z - (spacing / 2.0f) - eps;
        elem.aabbMaxX = point.x + (spacing / 2.0f) + eps;
        elem.aabbMaxY = point.y + (spacing / 2.0f) + eps;
        elem.aabbMaxZ = point.z + (spacing / 2.0f) + eps;
        aabbs.push_back(elem);
    }
    
    return aabbs;
}

void engine::LBVH::generateElements(std::vector<Element> &elements, engine::AABB *extent) {
  // TODO
}

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

  }
  catch (const std::runtime_error& e)
  {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }

  try
  {
    float spacing = 0.1f;
    float maxAbsValue = 1.0f;
    float eps = 0.01f;

    std::vector<Point3D> gridPoints = generateRegularGrid(maxAbsValue, spacing);

    std::vector<Point3D> trimmedGridPoint = trimPointsWithPositiveSDF(gridPoints, siren_nn);

    std::vector<Element> aabbs = generateAABBs(trimmedGridPoint, spacing, eps);

    // for (const auto& elem : aabbs) {
    //   std::cout << "Primitive Index: " << elem.primitiveIdx << std::endl;
    //   std::cout << "AABB Min: (" << elem.aabbMinX << ", " << elem.aabbMinY << ", " << elem.aabbMinZ << ")" << std::endl;
    //   std::cout << "AABB Max: (" << elem.aabbMaxX << ", " << elem.aabbMaxY << ", " << elem.aabbMaxZ << ")" << std::endl;
    //   std::cout << std::endl;
    // }

#ifdef RESOURCE_DIRECTORY_PATH
    engine::Paths::m_resourceDirectoryPath = RESOURCE_DIRECTORY_PATH;
#endif
    engine::GPUContext gpu(engine::Queues::QueueFamilies::COMPUTE_FAMILY | engine::Queues::TRANSFER_FAMILY);
    gpu.init();
    auto app = std::make_shared<engine::LBVH>();
    app->execute(&gpu);
    gpu.shutdown();
  }
  catch (const std::runtime_error& e)
  {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

