#pragma once
#include <vector>

namespace nsdf
{
  struct SDFArch
  {
    struct Shape {
      Shape (int i, int o) : input(i), output(o) {}
      int input = 0;
      int output = 0;
    };

    std::vector<Shape> layers;

    bool from_file(const char *filename);
  };

  struct SIREN
  {
    typedef std::vector<float> Row;
    typedef std::vector<Row> Matrix;

    std::vector<Matrix> As;
    std::vector<Row> bs;

    Matrix mat_mul(const Matrix& A, const Matrix& B);

    Matrix mat_sum(const Matrix& A, const Matrix& B);

    Matrix row_to_column(const Row& row);

    Matrix sin(const Matrix& input, float w0 = 30);

    float forward(const std::vector<float>& point);

    bool from_file(const char *filename, const SDFArch& arch);
  };

  struct SDFTest
  {
    int points_count;
    std::vector<float> points; // pos_0.x, pos_0.y, pos_0.z, pos_1.x, ...
    std::vector<float> etalon_dists; // dist_0, dist_0, dist_0, dist_1, ...

    bool from_file(const char *filename);
  };

}

