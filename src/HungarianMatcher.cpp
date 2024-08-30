#include <HungarianMatcher.h>

void HungarianMatcher::compute(const cv::Mat& cost_matrix, std::vector<int>& assignment) {
    int rows = cost_matrix.rows;
    int cols = cost_matrix.cols;

    cv::Mat matrix = cost_matrix.clone();
    std::cout << "hungary::1" << " " << rows << " " << cols << std::endl;
    // Step 1: Subtract row minima
    for (int i = 0; i < rows; ++i) {
        double min_val;
        cv::minMaxLoc(matrix.row(i), &min_val);
        matrix.row(i) -= min_val;
    }
    // Step 2: Subtract column minima
    for (int j = 0; j < cols; ++j) {
        double min_val;
        cv::minMaxLoc(matrix.col(j), &min_val);
        matrix.col(j) -= min_val;
    }
    std::cout << "hungary::3" << std::endl;
    std::vector<int> row_cover(rows, 0);
    std::vector<int> col_cover(cols, 0);
    assignment = std::vector<int>(rows, -1);

    // Step 3: Cover all zeros with minimum number of lines
    coverZeros(matrix, row_cover, col_cover);
    std::cout << "hungary::4" << std::endl;
    // Step 4: Create additional zeros
    while (std::accumulate(row_cover.begin(), row_cover.end(), 0) +
        std::accumulate(col_cover.begin(), col_cover.end(), 0) < rows) {
        createAdditionalZeros(matrix, row_cover, col_cover);
        coverZeros(matrix, row_cover, col_cover);
    }
    std::cout << "hungary::5" << std::endl;
    std::cout << matrix << std::endl;

    // Step 5: Find assignments
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (row_cover[i] == 0 && col_cover[j] == 0)
            {
                std::cout << "assignment = " <<matrix.at<int>(i,j)<<" = " << i << " " << j << std::endl;
            }
            if (matrix.at<int>(i, j) == 0 && row_cover[i] == 0 && col_cover[j] == 0) {
                assignment[i] = j;
                row_cover[i] = 1;
                col_cover[j] = 1;
                break;
            }
        }
    }
    std::cout <<"aa = " << assignment.size() << std::endl;
    std::cout << "hungary::6" << std::endl;
}