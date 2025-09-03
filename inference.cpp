#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <numeric>

int main() {
    std::ifstream file("../predictions/predicted_stock.csv"); // CSV exported from Python
    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    std::vector<double> open_prices;
    std::vector<double> close_prices;
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        int col = 0;
        double open, close;

        while (std::getline(ss, token, ',')) {
            if (col == 0) open = std::stod(token);  // Open
            if (col == 1) close = std::stod(token); // Close
            col++;
        }
        open_prices.push_back(open);
        close_prices.push_back(close);
    }
    file.close();

    // Compute average predicted Open and Close
    double avg_open = std::accumulate(open_prices.begin(), open_prices.end(), 0.0) / open_prices.size();
    double avg_close = std::accumulate(close_prices.begin(), close_prices.end(), 0.0) / close_prices.size();

    std::cout << "Average Predicted Open: " << avg_open << "\n";
    std::cout << "Average Predicted Close: " << avg_close << "\n";

    return 0;
}
