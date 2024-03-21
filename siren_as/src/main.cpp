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

// NOTE: different from engine::AABB (no glm::*)
struct AABB {
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

std::vector<AABB> generateAABBs(const std::vector<Point3D>& gridPoints, float spacing, float eps) {
    std::vector<AABB> aabbs;
    
    for (const auto& point : gridPoints) {
        AABB elem;
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

void engine::LBVH::generateElements(std::vector<engine::LBVH::Element> &elements, engine::AABB *extent) {
  float spacing = 0.1f;
  float maxAbsValue = 1.0f;
  float eps = 0.01f;

  nsdf::SDFArch sdf_arch;
  sdf_arch.from_file("siren_as/res/sdf1_arch.txt");
  nsdf::SIREN siren_nn;
  siren_nn.from_file("siren_as/res/sdf1_weights.bin", sdf_arch);

  std::vector<Point3D> gridPoints = generateRegularGrid(maxAbsValue, spacing);

  std::vector<Point3D> trimmedGridPoint = trimPointsWithPositiveSDF(gridPoints, siren_nn);

  std::vector<::AABB> aabbs = generateAABBs(trimmedGridPoint, spacing, eps);

  uint32_t primitiveIndex = 0;
  for (const auto& elem : aabbs) {
    elements.push_back({primitiveIndex, elem.aabbMinX, elem.aabbMinY, elem.aabbMinZ, elem.aabbMaxX, elem.aabbMaxY, elem.aabbMaxZ});
    primitiveIndex++;
    extent->expand({elem.aabbMinX, elem.aabbMinY, elem.aabbMinZ});
    extent->expand({elem.aabbMaxX, elem.aabbMaxY, elem.aabbMaxZ});
  }
}

int main (int argc, char *argv[])
{
  try
  {
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

