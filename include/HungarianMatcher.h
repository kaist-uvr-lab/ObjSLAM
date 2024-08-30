#ifndef OBJECT_SLAM_HUNGARIANMATCHER_H
#define OBJECT_SLAM_HUNGARIANMATCHER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
#include <numeric>

class HungarianMatcher {
public:
    static void compute(const cv::Mat& cost_matrix, std::vector<int>& assignment);

private:
    static void coverZeros(const cv::Mat& matrix, std::vector<int>& row_cover, std::vector<int>& col_cover) {
        int rows = matrix.rows;
        int cols = matrix.cols;

        std::fill(row_cover.begin(), row_cover.end(), 0);
        std::fill(col_cover.begin(), col_cover.end(), 0);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
            {
                if (matrix.at<int>(i, j) == 0 && row_cover[i] == 0 && col_cover[j] == 0) {
                    row_cover[i] = 1;
                    col_cover[j] = 1;
                }
            }
        }
    }

    static void createAdditionalZeros(cv::Mat& matrix, const std::vector<int>& row_cover, const std::vector<int>& col_cover) {
        int rows = matrix.rows; 
        int cols = matrix.cols;

        int min_val = std::numeric_limits<int>::max();

        for (int i = 0; i < rows; ++i) {
            if (row_cover[i] == 0) {
                for (int j = 0; j < cols; ++j) {
                    if (col_cover[j] == 0) {
                        min_val = std::min(min_val, matrix.at<int>(i, j));
                    }
                }
            }
        }

        for (int i = 0; i < rows; ++i) {
            if (row_cover[i] == 0) {
                for (int j = 0; j < cols; ++j) {
                    if (col_cover[j] == 0) {
                        matrix.at<int>(i, j) -= min_val;
                    }
                }
            }
        }

        for (int i = 0; i < rows; ++i) {
            if (row_cover[i] == 1) {
                for (int j = 0; j < cols; ++j) {
                    if (col_cover[j] == 1) {
                        matrix.at<int>(i, j) += min_val;
                    }
                }
            }
        }
    }
};

#endif