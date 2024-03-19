#include <cassert>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <vector>
#include <cmath>

#include "nsdf.h"

namespace nsdf
{
  void file_check(FILE* f, const char* filename) {
    long current_position = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long remaining_bytes = file_size - current_position;
    if (file_size != current_position) {
      printf("[Warn] Осталось %ld байтов до конца файла %s.\n", remaining_bytes, filename);
    }
  }

  bool SDFArch::from_file(const char *filename)
  {
    FILE *f = fopen(filename, "r");
    if (!f)
    {
      fprintf(stderr, "failed to open file %s. Errno %d\n",filename, (int)errno);
      return false;
    }

    int input = 3;
    int output = 0;
    fscanf(f, "Dense input shape (3) output shape (%d)\n", &output);
    layers.emplace_back(input, output);

    do {
      fscanf(f, "Sin input shape (%d) output shape (%d)\n", &input, &output);
      // layers.emplace_back(input, output);
      fscanf(f, "Dense input shape (%d) output shape (%d)\n", &input, &output);
      layers.emplace_back(input, output);
    } while (output != 1);

    file_check(f, filename);
    int res = fclose(f);
    if (res != 0)
    {
      fprintf(stderr, "failed to close file %s. fclose returned %d\n",filename, res);
      return false;
    }
    return true;
  }

  SIREN::Matrix SIREN::mat_mul(const SIREN::Matrix& A, const SIREN::Matrix& B) {
    size_t n = A.size();
    size_t m = A[0].size();
    size_t p = B[0].size();
    SIREN::Matrix C(n, std::vector<float>(p, 0.0));

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < p; ++j) {
        for (size_t k = 0; k < m; ++k) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }

    return C;
  }

  SIREN::Matrix SIREN::mat_sum(const SIREN::Matrix& A, const SIREN::Matrix& B) {
    size_t n = A.size();
    size_t p = A[0].size();
    SIREN::Matrix C(n, std::vector<float>(p, 0.0));

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < p; ++j) {
        C[i][j] = A[i][j] + B[i][j];
      }
    }

    return C;
  }

  SIREN::Matrix SIREN::row_to_column(const Row& row) {
    SIREN::Matrix transposed_row(row.size(), std::vector<float>(1));
    for (size_t i = 0; i < row.size(); i++) {
      transposed_row[i][0] = row[i];
    }
    return transposed_row;
  }

  SIREN::Matrix SIREN::sin(const SIREN::Matrix& input, float w0) {
    size_t n = input.size();
    size_t p = input[0].size();
    SIREN::Matrix C(n, std::vector<float>(p, 0.0));

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < p; ++j) {
        C[i][j] = std::sin(w0 * input[i][j]);
      }
    }

    return C;
  }

  float SIREN::forward(const std::vector<float>& point)
  {
    assert(point.size() == 3);
    SIREN::Matrix current = row_to_column(point);

    // 1
    current = mat_mul(As[0], current);
    current = mat_sum(current, row_to_column(bs[0]));

    // 2
    current = sin(current);

    // 3
    current = mat_mul(As[1], current);
    current = mat_sum(current, row_to_column(bs[1]));

    // 4
    current = sin(current);

    // 5
    current = mat_mul(As[2], current);
    current = mat_sum(current, row_to_column(bs[2]));

    // 6
    current = sin(current);

    // 7
    current = mat_mul(As[3], current);
    current = mat_sum(current, row_to_column(bs[3]));

    assert(current.size() == 1);
    assert(current[0].size() == 1);
    return current[0][0];
  }

  bool SIREN::from_file(const char *filename, const SDFArch& arch)
  {
    FILE *f = fopen(filename, "r");
    if (!f)
    {
      fprintf(stderr, "failed to open file %s. Errno %d\n",filename, (int)errno);
      return false;
    }

    size_t layers_count = arch.layers.size();

    As.resize(layers_count);
    bs.resize(layers_count);

    for (size_t i = 0; i < layers_count; i++) {
      auto[input, output] = arch.layers[i];
      As[i].resize(output);
      for (auto& row : As[i]) {
        row.resize(input);
        fread(row.data(), sizeof(float), input, f);
      }

      bs[i].resize(output);
      fread(bs[i].data(), sizeof(float), output, f);
    }

    file_check(f, filename);
    int res = fclose(f);
    if (res != 0)
    {
      fprintf(stderr, "failed to close file %s. fclose returned %d\n",filename, res);
      return false;
    }
    return true;
  }

  bool SDFTest::from_file(const char *filename)
  {
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
      fprintf(stderr, "failed to open file %s. Errno %d\n",filename, (int)errno);
      return false;
    }

    fread(&points_count, sizeof(int), 1, f);

    points.resize(points_count * 3);
    fread(points.data(), sizeof(float), points_count * 3, f);
    etalon_dists.resize(points_count);
    fread(etalon_dists.data(), sizeof(float), points_count, f);

    file_check(f, filename);
    int res = fclose(f);
    if (res != 0)
    {
      fprintf(stderr, "failed to close file %s. fclose returned %d\n",filename, res);
      return false;
    }
    return true;
  }
}

