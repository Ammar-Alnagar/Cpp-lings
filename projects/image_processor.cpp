/*
 * Project 3: Image Processing Pipeline
 * 
 * This project implements a modular image processing pipeline that demonstrates:
 * - Advanced OOP with inheritance and polymorphism
 * - Templates and generic programming
 * - STL algorithms and containers
 * - Smart pointers and RAII
 * - Concurrency and multithreading
 * - Function objects and lambdas
 * - Modern C++ features
 * 
 * The system processes images through a chain of filters applied in parallel.
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <thread>
#include <mutex>
#include <future>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;

// TODO: Define Image class to represent an image
class Image {
private:
    vector<vector<int>> pixels;  // Using int for simplicity, could be RGB values
    size_t width, height;
    string name;
    
public:
    Image(size_t w, size_t h, const string& n = "unnamed") 
        : width(w), height(h), name(n) {
        pixels.resize(height, vector<int>(width, 0));
        cout << "Image '" << name << "' created: " << width << "x" << height << endl;
    }
    
    // Constructor from existing pixel data
    Image(const vector<vector<int>>& data, const string& n = "unnamed")
        : pixels(data), width(data[0].size()), height(data.size()), name(n) {
        cout << "Image '" << name << "' created from data: " << width << "x" << height << endl;
    }
    
    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }
    const string& getName() const { return name; }
    
    // Access pixel with bounds checking
    int& at(size_t x, size_t y) {
        if (x >= width || y >= height) {
            throw out_of_range("Pixel coordinates out of bounds");
        }
        return pixels[y][x];
    }
    
    const int& at(size_t x, size_t y) const {
        if (x >= width || y >= height) {
            throw out_of_range("Pixel coordinates out of bounds");
        }
        return pixels[y][x];
    }
    
    // Get entire pixel grid
    const vector<vector<int>>& getPixels() const { return pixels; }
    
    // Set all pixels to a value
    void fill(int value) {
        for (auto& row : pixels) {
            fill(row.begin(), row.end(), value);
        }
    }
    
    // Apply a function to each pixel
    void transform(const function<int(int)>& func) {
        for (auto& row : pixels) {
            for (auto& pixel : row) {
                pixel = func(pixel);
            }
        }
    }
    
    // Display image (simplified - just show dimensions and some sample values)
    void display() const {
        cout << "Image: " << name << " (" << width << "x" << height << ")" << endl;
        cout << "Sample pixels (top-left 5x5):" << endl;
        
        size_t displayHeight = min(height, static_cast<size_t>(5));
        size_t displayWidth = min(width, static_cast<size_t>(5));
        
        for (size_t y = 0; y < displayHeight; y++) {
            for (size_t x = 0; x < displayWidth; x++) {
                cout << setw(4) << pixels[y][x] << " ";
            }
            cout << endl;
        }
        if (width > 5 || height > 5) {
            cout << "... (truncated)" << endl;
        }
    }
    
    // Create a copy of the image
    Image clone() const {
        return Image(pixels, name + "_clone");
    }
};

// TODO: Define abstract base class for image filters
class ImageFilter {
protected:
    string name;
    
public:
    explicit ImageFilter(const string& filterName) : name(filterName) {}
    virtual ~ImageFilter() = default;
    
    const string& getName() const { return name; }
    
    // Pure virtual function for applying the filter
    virtual unique_ptr<Image> apply(const Image& input) const = 0;
    
    // Virtual function for getting filter information
    virtual string getInfo() const {
        return "Filter: " + name;
    }
};

// TODO: Implement concrete filter classes
class GrayscaleFilter : public ImageFilter {
public:
    GrayscaleFilter() : ImageFilter("Grayscale") {}
    
    unique_ptr<Image> apply(const Image& input) const override {
        cout << "Applying " << name << " filter..." << endl;
        
        auto result = make_unique<Image>(input.getWidth(), input.getHeight(), 
                                        input.getName() + "_grayscale");
        
        for (size_t y = 0; y < input.getHeight(); y++) {
            for (size_t x = 0; x < input.getWidth(); x++) {
                int pixel = input.at(x, y);
                // Simplified grayscale: average of RGB channels (here just using the single value)
                int gray = (pixel + 128) / 2;  // Adjust for demonstration
                result->at(x, y) = gray;
            }
        }
        
        this_thread::sleep_for(chrono::milliseconds(10)); // Simulate processing time
        return result;
    }
    
    string getInfo() const override {
        return "Grayscale Filter: Converts image to grayscale";
    }
};

class BlurFilter : public ImageFilter {
private:
    int radius;
    
public:
    explicit BlurFilter(int r = 1) : ImageFilter("Blur"), radius(r) {}
    
    unique_ptr<Image> apply(const Image& input) const override {
        cout << "Applying " << name << " filter (radius: " << radius << ")..." << endl;
        
        auto result = make_unique<Image>(input.getWidth(), input.getHeight(), 
                                        input.getName() + "_blur");
        
        for (size_t y = 0; y < input.getHeight(); y++) {
            for (size_t x = 0; x < input.getWidth(); x++) {
                int sum = 0;
                int count = 0;
                
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int nx = static_cast<int>(x) + dx;
                        int ny = static_cast<int>(y) + dy;
                        
                        if (nx >= 0 && nx < static_cast<int>(input.getWidth()) &&
                            ny >= 0 && ny < static_cast<int>(input.getHeight())) {
                            sum += input.at(nx, ny);
                            count++;
                        }
                    }
                }
                
                result->at(x, y) = sum / count;
            }
        }
        
        this_thread::sleep_for(chrono::milliseconds(20)); // Simulate processing time
        return result;
    }
    
    string getInfo() const override {
        return "Blur Filter: Applies Gaussian blur with radius " + to_string(radius);
    }
};

class EdgeDetectionFilter : public ImageFilter {
public:
    EdgeDetectionFilter() : ImageFilter("Edge Detection") {}
    
    unique_ptr<Image> apply(const Image& input) const override {
        cout << "Applying " << name << " filter..." << endl;
        
        auto result = make_unique<Image>(input.getWidth(), input.getHeight(), 
                                        input.getName() + "_edges");
        
        // Simplified edge detection using Sobel-like operator
        for (size_t y = 1; y < input.getHeight() - 1; y++) {
            for (size_t x = 1; x < input.getWidth() - 1; x++) {
                int gx = (input.at(x-1, y-1) + 2*input.at(x-1, y) + input.at(x-1, y+1)) -
                         (input.at(x+1, y-1) + 2*input.at(x+1, y) + input.at(x+1, y+1));
                
                int gy = (input.at(x-1, y-1) + 2*input.at(x, y-1) + input.at(x+1, y-1)) -
                         (input.at(x-1, y+1) + 2*input.at(x, y+1) + input.at(x+1, y+1));
                
                int magnitude = static_cast<int>(sqrt(gx*gx + gy*gy));
                result->at(x, y) = min(255, magnitude);  // Clamp to valid range
            }
        }
        
        this_thread::sleep_for(chrono::milliseconds(30)); // Simulate processing time
        return result;
    }
    
    string getInfo() const override {
        return "Edge Detection Filter: Detects edges in the image";
    }
};

class BrightnessFilter : public ImageFilter {
private:
    int brightnessAdjustment;
    
public:
    explicit BrightnessFilter(int adjustment) 
        : ImageFilter("Brightness"), brightnessAdjustment(adjustment) {}
    
    unique_ptr<Image> apply(const Image& input) const override {
        cout << "Applying " << name << " filter (adjustment: " << brightnessAdjustment << ")..." << endl;
        
        auto result = make_unique<Image>(input.getWidth(), input.getHeight(), 
                                        input.getName() + "_bright");
        
        for (size_t y = 0; y < input.getHeight(); y++) {
            for (size_t x = 0; x < input.getWidth(); x++) {
                int newPixel = input.at(x, y) + brightnessAdjustment;
                // Clamp to valid range [0, 255]
                newPixel = max(0, min(255, newPixel));
                result->at(x, y) = newPixel;
            }
        }
        
        this_thread::sleep_for(chrono::milliseconds(5)); // Simulate processing time
        return result;
    }
    
    string getInfo() const override {
        return "Brightness Filter: Adjusts image brightness by " + to_string(brightnessAdjustment);
    }
};

// TODO: Create ImageProcessor class to manage filter pipelines
class ImageProcessor {
private:
    vector<unique_ptr<ImageFilter>> filters;
    mutable mutex processingMutex;  // For thread safety during processing
    
public:
    void addFilter(unique_ptr<ImageFilter> filter) {
        filters.push_back(move(filter));
    }
    
    void addFilter(const ImageFilter& filter) {
        // Create a copy of the filter
        if (dynamic_cast<const GrayscaleFilter*>(&filter)) {
            addFilter(make_unique<GrayscaleFilter>());
        } else if (dynamic_cast<const BlurFilter*>(&filter)) {
            auto blur = static_cast<const BlurFilter*>(&filter);
            addFilter(make_unique<BlurFilter>(blur->getRadius()));  // Assuming getRadius() exists
        } else if (dynamic_cast<const EdgeDetectionFilter*>(&filter)) {
            addFilter(make_unique<EdgeDetectionFilter>());
        } else if (dynamic_cast<const BrightnessFilter*>(&filter)) {
            auto bright = static_cast<const BrightnessFilter*>(&filter);
            addFilter(make_unique<BrightnessFilter>(bright->getAdjustment()));  // Assuming getAdjustment() exists
        }
    }
    
    // TODO: Implement sequential processing
    unique_ptr<Image> processSequential(const Image& input) const {
        cout << "\n=== Sequential Processing ===" << endl;
        
        unique_ptr<Image> current = make_unique<Image>(input.clone());
        
        for (const auto& filter : filters) {
            cout << "Applying filter: " << filter->getName() << endl;
            current = filter->apply(*current);
        }
        
        return current;
    }
    
    // TODO: Implement parallel processing
    unique_ptr<Image> processParallel(const Image& input) const {
        cout << "\n=== Parallel Processing ===" << endl;
        
        if (filters.empty()) {
            return make_unique<Image>(input.clone());
        }
        
        // For parallel processing, we'll apply each filter to a copy of the original image
        // and then combine the results (this is a simplified approach)
        vector<future<unique_ptr<Image>>> futures;
        
        for (const auto& filter : filters) {
            cout << "Starting parallel filter: " << filter->getName() << endl;
            
            // Create a copy of input for each filter
            Image inputCopy = input.clone();
            
            futures.push_back(async(launch::async, [filter, inputCopy]() {
                return filter->apply(inputCopy);
            }));
        }
        
        // Wait for all filters to complete
        vector<unique_ptr<Image>> results;
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        // For this example, we'll just return the result of the last filter
        // In a real system, you'd have logic to combine the results appropriately
        return move(results.back());
    }
    
    // TODO: Implement pipeline processing (each filter processes the output of the previous)
    unique_ptr<Image> processPipeline(const Image& input) const {
        cout << "\n=== Pipeline Processing ===" << endl;
        
        if (filters.empty()) {
            return make_unique<Image>(input.clone());
        }
        
        // Apply filters sequentially but with potential for optimization
        unique_ptr<Image> current = make_unique<Image>(input.clone());
        
        for (size_t i = 0; i < filters.size(); i++) {
            cout << "Applying filter " << (i+1) << "/" << filters.size() 
                 << ": " << filters[i]->getName() << endl;
            
            auto start = chrono::high_resolution_clock::now();
            current = filters[i]->apply(*current);
            auto end = chrono::high_resolution_clock::now();
            
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            cout << "  Filter took " << duration.count() << " ms" << endl;
        }
        
        return current;
    }
    
    // TODO: Implement batch processing
    vector<unique_ptr<Image>> processBatch(const vector<Image>& inputs) const {
        cout << "\n=== Batch Processing ===" << endl;
        
        vector<unique_ptr<Image>> results;
        results.reserve(inputs.size());
        
        for (size_t i = 0; i < inputs.size(); i++) {
            cout << "Processing image " << (i+1) << "/" << inputs.size() << endl;
            results.push_back(processPipeline(inputs[i]));
        }
        
        return results;
    }
    
    // TODO: Implement parallel batch processing
    vector<unique_ptr<Image>> processBatchParallel(const vector<Image>& inputs) const {
        cout << "\n=== Parallel Batch Processing ===" << endl;
        
        vector<future<unique_ptr<Image>>> futures;
        futures.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            futures.push_back(async(launch::async, [this, input]() {
                return processPipeline(input);
            }));
        }
        
        vector<unique_ptr<Image>> results;
        results.reserve(inputs.size());
        
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        return results;
    }
    
    // TODO: Display pipeline information
    void displayPipeline() const {
        cout << "\n=== Current Processing Pipeline ===" << endl;
        if (filters.empty()) {
            cout << "No filters in pipeline" << endl;
            return;
        }
        
        for (size_t i = 0; i < filters.size(); i++) {
            cout << (i+1) << ". " << filters[i]->getInfo() << endl;
        }
    }
    
    // TODO: Clear pipeline
    void clearPipeline() {
        filters.clear();
    }
    
    // TODO: Get filter count
    size_t getFilterCount() const {
        return filters.size();
    }
    
    // TODO: Remove filter by index
    bool removeFilter(size_t index) {
        if (index >= filters.size()) {
            return false;
        }
        filters.erase(filters.begin() + index);
        return true;
    }
    
    // TODO: Get filter by name
    ImageFilter* getFilterByName(const string& name) const {
        for (const auto& filter : filters) {
            if (filter->getName() == name) {
                return filter.get();
            }
        }
        return nullptr;
    }
};

// TODO: Create a filter factory for easy filter creation
class FilterFactory {
public:
    static unique_ptr<ImageFilter> createFilter(const string& type, const vector<int>& params = {}) {
        if (type == "grayscale") {
            return make_unique<GrayscaleFilter>();
        } else if (type == "blur") {
            int radius = params.empty() ? 1 : params[0];
            return make_unique<BlurFilter>(radius);
        } else if (type == "edge") {
            return make_unique<EdgeDetectionFilter>();
        } else if (type == "brightness") {
            int adjustment = params.empty() ? 10 : params[0];
            return make_unique<BrightnessFilter>(adjustment);
        } else {
            throw invalid_argument("Unknown filter type: " + type);
        }
    }
    
    static vector<string> getAvailableFilters() {
        return {"grayscale", "blur", "edge", "brightness"};
    }
};

int main() {
    cout << "=== Image Processing Pipeline ===" << endl;
    
    // TODO: Create sample images
    Image original(100, 100, "Original");
    
    // Fill with some pattern for visualization
    for (size_t y = 0; y < original.getHeight(); y++) {
        for (size_t x = 0; x < original.getWidth(); x++) {
            original.at(x, y) = (x + y) % 256;  // Simple gradient pattern
        }
    }
    
    cout << "Created original image:" << endl;
    original.display();
    
    // TODO: Create image processor and add filters
    ImageProcessor processor;
    
    // Add filters using the factory
    processor.addFilter(FilterFactory::createFilter("grayscale"));
    processor.addFilter(FilterFactory::createFilter("blur", {2}));
    processor.addFilter(FilterFactory::createFilter("brightness", {20}));
    processor.addFilter(FilterFactory::createFilter("edge"));
    
    processor.displayPipeline();
    
    // TODO: Process image sequentially
    auto sequentialResult = processor.processSequential(original);
    cout << "\nSequential processing result:" << endl;
    sequentialResult->display();
    
    // TODO: Process image in pipeline mode
    auto pipelineResult = processor.processPipeline(original);
    cout << "\nPipeline processing result:" << endl;
    pipelineResult->display();
    
    // TODO: Process image in parallel mode
    auto parallelResult = processor.processParallel(original);
    cout << "\nParallel processing result:" << endl;
    parallelResult->display();
    
    // TODO: Create multiple sample images for batch processing
    vector<Image> batchImages;
    batchImages.emplace_back(50, 50, "Batch1");
    batchImages.emplace_back(60, 60, "Batch2");
    batchImages.emplace_back(70, 70, "Batch3");
    
    // Fill batch images with different patterns
    for (size_t i = 0; i < batchImages.size(); i++) {
        for (size_t y = 0; y < batchImages[i].getHeight(); y++) {
            for (size_t x = 0; x < batchImages[i].getWidth(); x++) {
                batchImages[i].at(x, y) = (i * 50 + x + y) % 256;
            }
        }
    }
    
    // TODO: Process batch sequentially
    cout << "\n=== Batch Processing Demo ===" << endl;
    auto batchResults = processor.processBatch(batchImages);
    
    cout << "Batch processing results:" << endl;
    for (size_t i = 0; i < batchResults.size(); i++) {
        cout << "Result " << (i+1) << ":" << endl;
        batchResults[i]->display();
    }
    
    // TODO: Process batch in parallel
    cout << "\n=== Parallel Batch Processing Demo ===" << endl;
    auto parallelBatchResults = processor.processBatchParallel(batchImages);
    
    cout << "Parallel batch processing results:" << endl;
    for (size_t i = 0; i < parallelBatchResults.size(); i++) {
        cout << "Parallel Result " << (i+1) << ":" << endl;
        parallelBatchResults[i]->display();
    }
    
    // TODO: Performance comparison
    cout << "\n=== Performance Comparison ===" << endl;
    
    Image perfTestImage(200, 200, "PerformanceTest");
    for (size_t y = 0; y < perfTestImage.getHeight(); y++) {
        for (size_t x = 0; x < perfTestImage.getWidth(); x++) {
            perfTestImage.at(x, y) = (x * y) % 256;
        }
    }
    
    // Time sequential processing
    auto start = chrono::high_resolution_clock::now();
    auto seqResult = processor.processSequential(perfTestImage);
    auto seqEnd = chrono::high_resolution_clock::now();
    auto seqDuration = chrono::duration_cast<chrono::milliseconds>(seqEnd - start);
    
    // Time pipeline processing
    start = chrono::high_resolution_clock::now();
    auto pipeResult = processor.processPipeline(perfTestImage);
    auto pipeEnd = chrono::high_resolution_clock::now();
    auto pipeDuration = chrono::duration_cast<chrono::milliseconds>(pipeEnd - start);
    
    // Time parallel processing
    start = chrono::high_resolution_clock::now();
    auto parResult = processor.processParallel(perfTestImage);
    auto parEnd = chrono::high_resolution_clock::now();
    auto parDuration = chrono::duration_cast<chrono::milliseconds>(parEnd - start);
    
    cout << "Sequential processing time: " << seqDuration.count() << " ms" << endl;
    cout << "Pipeline processing time: " << pipeDuration.count() << " ms" << endl;
    cout << "Parallel processing time: " << parDuration.count() << " ms" << endl;
    
    // TODO: Filter management
    cout << "\n=== Filter Management ===" << endl;
    cout << "Number of filters: " << processor.getFilterCount() << endl;
    
    // Try to get a specific filter
    ImageFilter* grayscaleFilter = processor.getFilterByName("Grayscale");
    if (grayscaleFilter) {
        cout << "Found filter: " << grayscaleFilter->getInfo() << endl;
    }
    
    // Remove a filter
    bool removed = processor.removeFilter(1);  // Remove blur filter
    cout << "Filter removed: " << (removed ? "Yes" : "No") << endl;
    cout << "Number of filters after removal: " << processor.getFilterCount() << endl;
    
    processor.displayPipeline();
    
    // TODO: Create a complex pipeline with custom lambda filters
    cout << "\n=== Custom Lambda Filters ===" << endl;
    
    // Create a custom filter using lambda and function wrapper
    class LambdaFilter : public ImageFilter {
    private:
        function<unique_ptr<Image>(const Image&)> filterFunc;
        
    public:
        LambdaFilter(const string& name, function<unique_ptr<Image>(const Image&)> func)
            : ImageFilter(name), filterFunc(move(func)) {}
        
        unique_ptr<Image> apply(const Image& input) const override {
            return filterFunc(input);
        }
    };
    
    // Add a custom noise filter
    auto noiseFilter = make_unique<LambdaFilter>("Noise", [](const Image& input) {
        cout << "Applying noise filter..." << endl;
        auto result = make_unique<Image>(input.getWidth(), input.getHeight(), 
                                        input.getName() + "_noise");
        
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(-10, 10);
        
        for (size_t y = 0; y < input.getHeight(); y++) {
            for (size_t x = 0; x < input.getWidth(); x++) {
                int noisyValue = input.at(x, y) + dis(gen);
                result->at(x, y) = max(0, min(255, noisyValue));  // Clamp to [0, 255]
            }
        }
        
        this_thread::sleep_for(chrono::milliseconds(15)); // Simulate processing time
        return result;
    });
    
    ImageProcessor customProcessor;
    customProcessor.addFilter(move(noiseFilter));
    customProcessor.addFilter(FilterFactory::createFilter("blur", {1}));
    
    auto customResult = customProcessor.processPipeline(perfTestImage);
    cout << "Custom pipeline result:" << endl;
    customResult->display();
    
    cout << "\nImage Processing Pipeline completed successfully!" << endl;
    
    return 0;
}