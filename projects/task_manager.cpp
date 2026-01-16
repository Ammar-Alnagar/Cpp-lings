/*
 * Project 1: Task Management System
 * 
 * This project implements a comprehensive task management system that demonstrates:
 * - Object-oriented programming principles
 * - STL containers and algorithms
 * - Memory management with smart pointers
 * - File I/O and serialization
 * - Exception handling
 * - Templates and generic programming
 * - RAII principles
 * 
 * The system allows users to create, manage, and track tasks with priorities, deadlines, and categories.
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <map>
#include <set>
#include <functional>
#include <chrono>
#include <ctime>

using namespace std;
using namespace std::chrono;

// TODO: Define TaskStatus enum
enum class TaskStatus {
    PENDING,
    IN_PROGRESS,
    COMPLETED,
    CANCELLED
};

// TODO: Define Priority enum
enum class Priority {
    LOW,
    MEDIUM,
    HIGH,
    URGENT
};

// TODO: Create Task class with proper encapsulation
class Task {
private:
    int id;
    string title;
    string description;
    TaskStatus status;
    Priority priority;
    string category;
    system_clock::time_point deadline;
    system_clock::time_point createdAt;
    system_clock::time_point updatedAt;

public:
    // TODO: Implement constructor
    Task(int taskId, const string& taskTitle, const string& taskDescription, 
         Priority taskPriority, const string& taskCategory, 
         system_clock::time_point taskDeadline)
        : id(taskId), title(taskTitle), description(taskDescription), 
          status(TaskStatus::PENDING), priority(taskPriority), 
          category(taskCategory), deadline(taskDeadline),
          createdAt(system_clock::now()), updatedAt(system_clock::now()) {}

    // TODO: Implement getter methods
    int getId() const { return id; }
    const string& getTitle() const { return title; }
    const string& getDescription() const { return description; }
    TaskStatus getStatus() const { return status; }
    Priority getPriority() const { return priority; }
    const string& getCategory() const { return category; }
    system_clock::time_point getDeadline() const { return deadline; }
    system_clock::time_point getCreatedAt() const { return createdAt; }
    system_clock::time_point getUpdatedAt() const { return updatedAt; }

    // TODO: Implement setter methods
    void setTitle(const string& newTitle) { 
        title = newTitle; 
        updatedAt = system_clock::now();
    }
    
    void setDescription(const string& newDescription) { 
        description = newDescription; 
        updatedAt = system_clock::now();
    }
    
    void setStatus(TaskStatus newStatus) { 
        status = newStatus; 
        updatedAt = system_clock::now();
    }
    
    void setPriority(Priority newPriority) { 
        priority = newPriority; 
        updatedAt = system_clock::now();
    }
    
    void setCategory(const string& newCategory) { 
        category = newCategory; 
        updatedAt = system_clock::now();
    }
    
    void setDeadline(system_clock::time_point newDeadline) { 
        deadline = newDeadline; 
        updatedAt = system_clock::now();
    }

    // TODO: Implement helper methods
    bool isOverdue() const {
        return system_clock::now() > deadline && status != TaskStatus::COMPLETED;
    }
    
    string statusToString() const {
        switch(status) {
            case TaskStatus::PENDING: return "Pending";
            case TaskStatus::IN_PROGRESS: return "In Progress";
            case TaskStatus::COMPLETED: return "Completed";
            case TaskStatus::CANCELLED: return "Cancelled";
            default: return "Unknown";
        }
    }
    
    string priorityToString() const {
        switch(priority) {
            case Priority::LOW: return "Low";
            case Priority::MEDIUM: return "Medium";
            case Priority::HIGH: return "High";
            case Priority::URGENT: return "Urgent";
            default: return "Unknown";
        }
    }
    
    // TODO: Implement display method
    void display() const {
        time_t deadlineTime = system_clock::to_time_t(deadline);
        cout << "ID: " << id 
             << ", Title: " << title 
             << ", Status: " << statusToString()
             << ", Priority: " << priorityToString()
             << ", Category: " << category
             << ", Deadline: " << put_time(localtime(&deadlineTime), "%Y-%m-%d %H:%M")
             << ", Overdue: " << (isOverdue() ? "Yes" : "No")
             << endl;
    }
};

// TODO: Create TaskManager class
class TaskManager {
private:
    vector<unique_ptr<Task>> tasks;
    int nextId;
    
public:
    TaskManager() : nextId(1) {}
    
    // TODO: Implement addTask method
    int addTask(const string& title, const string& description, 
                Priority priority, const string& category, 
                system_clock::time_point deadline) {
        auto newTask = make_unique<Task>(nextId++, title, description, priority, category, deadline);
        tasks.push_back(move(newTask));
        return nextId - 1;  // Return the ID of the newly created task
    }
    
    // TODO: Implement removeTask method
    bool removeTask(int taskId) {
        auto it = find_if(tasks.begin(), tasks.end(),
                         [taskId](const unique_ptr<Task>& task) {
                             return task->getId() == taskId;
                         });
        
        if (it != tasks.end()) {
            tasks.erase(it);
            return true;
        }
        return false;
    }
    
    // TODO: Implement findTask method
    Task* findTask(int taskId) {
        auto it = find_if(tasks.begin(), tasks.end(),
                         [taskId](const unique_ptr<Task>& task) {
                             return task->getId() == taskId;
                         });
        
        if (it != tasks.end()) {
            return it->get();
        }
        return nullptr;
    }
    
    // TODO: Implement getAllTasks method
    const vector<unique_ptr<Task>>& getAllTasks() const {
        return tasks;
    }
    
    // TODO: Implement getTasksByStatus method
    vector<Task*> getTasksByStatus(TaskStatus status) const {
        vector<Task*> result;
        for (const auto& task : tasks) {
            if (task->getStatus() == status) {
                result.push_back(task.get());
            }
        }
        return result;
    }
    
    // TODO: Implement getTasksByPriority method
    vector<Task*> getTasksByPriority(Priority priority) const {
        vector<Task*> result;
        for (const auto& task : tasks) {
            if (task->getPriority() == priority) {
                result.push_back(task.get());
            }
        }
        return result;
    }
    
    // TODO: Implement getTasksByCategory method
    vector<Task*> getTasksByCategory(const string& category) const {
        vector<Task*> result;
        for (const auto& task : tasks) {
            if (task->getCategory() == category) {
                result.push_back(task.get());
            }
        }
        return result;
    }
    
    // TODO: Implement getOverdueTasks method
    vector<Task*> getOverdueTasks() const {
        vector<Task*> result;
        for (const auto& task : tasks) {
            if (task->isOverdue()) {
                result.push_back(task.get());
            }
        }
        return result;
    }
    
    // TODO: Implement updateTaskStatus method
    bool updateTaskStatus(int taskId, TaskStatus newStatus) {
        Task* task = findTask(taskId);
        if (task) {
            task->setStatus(newStatus);
            return true;
        }
        return false;
    }
    
    // TODO: Implement displayAllTasks method
    void displayAllTasks() const {
        cout << "\n=== All Tasks ===" << endl;
        if (tasks.empty()) {
            cout << "No tasks found." << endl;
            return;
        }
        
        for (const auto& task : tasks) {
            task->display();
        }
    }
    
    // TODO: Implement displayTasksByStatus method
    void displayTasksByStatus(TaskStatus status) const {
        auto tasksByStatus = getTasksByStatus(status);
        cout << "\n=== Tasks with status: " << 
                (status == TaskStatus::PENDING ? "Pending" :
                 status == TaskStatus::IN_PROGRESS ? "In Progress" :
                 status == TaskStatus::COMPLETED ? "Completed" : "Cancelled") 
             << " ===" << endl;
                 
        if (tasksByStatus.empty()) {
            cout << "No tasks found with this status." << endl;
            return;
        }
        
        for (const auto& task : tasksByStatus) {
            task->display();
        }
    }
    
    // TODO: Implement saveToFile method
    void saveToFile(const string& filename) const {
        ofstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Could not open file for writing: " + filename);
        }
        
        for (const auto& task : tasks) {
            time_t deadlineTime = system_clock::to_time_t(task->getDeadline());
            file << task->getId() << ","
                 << task->getTitle() << ","
                 << task->getDescription() << ","
                 << static_cast<int>(task->getStatus()) << ","
                 << static_cast<int>(task->getPriority()) << ","
                 << task->getCategory() << ","
                 << deadlineTime << "\n";
        }
        cout << "Tasks saved to " << filename << endl;
    }
    
    // TODO: Implement loadFromFile method
    void loadFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Could not open file for reading: " + filename);
        }
        
        tasks.clear();  // Clear existing tasks
        string line;
        
        while (getline(file, line)) {
            stringstream ss(line);
            string field;
            vector<string> fields;
            
            while (getline(ss, field, ',')) {
                fields.push_back(field);
            }
            
            if (fields.size() >= 7) {  // Ensure we have enough fields
                int id = stoi(fields[0]);
                string title = fields[1];
                string description = fields[2];
                TaskStatus status = static_cast<TaskStatus>(stoi(fields[3]));
                Priority priority = static_cast<Priority>(stoi(fields[4]));
                string category = fields[5];
                time_t deadlineTime = stoll(fields[6]);
                
                auto deadline = system_clock::from_time_t(deadlineTime);
                
                auto task = make_unique<Task>(id, title, description, priority, category, deadline);
                task->setStatus(status);  // Set the loaded status
                tasks.push_back(move(task));
                
                // Update nextId if necessary
                if (id >= nextId) {
                    nextId = id + 1;
                }
            }
        }
        cout << "Tasks loaded from " << filename << endl;
    }
    
    // TODO: Implement searchTasks method
    vector<Task*> searchTasks(const string& searchTerm) const {
        vector<Task*> result;
        for (const auto& task : tasks) {
            if (task->getTitle().find(searchTerm) != string::npos ||
                task->getDescription().find(searchTerm) != string::npos ||
                task->getCategory().find(searchTerm) != string::npos) {
                result.push_back(task.get());
            }
        }
        return result;
    }
    
    // TODO: Implement getStatistics method
    void getStatistics() const {
        cout << "\n=== Task Statistics ===" << endl;
        cout << "Total tasks: " << tasks.size() << endl;
        
        map<TaskStatus, int> statusCount;
        map<Priority, int> priorityCount;
        set<string> categories;
        
        for (const auto& task : tasks) {
            statusCount[task->getStatus()]++;
            priorityCount[task->getPriority()]++;
            categories.insert(task->getCategory());
        }
        
        cout << "Status distribution:" << endl;
        cout << "  Pending: " << statusCount[TaskStatus::PENDING] << endl;
        cout << "  In Progress: " << statusCount[TaskStatus::IN_PROGRESS] << endl;
        cout << "  Completed: " << statusCount[TaskStatus::COMPLETED] << endl;
        cout << "  Cancelled: " << statusCount[TaskStatus::CANCELLED] << endl;
        
        cout << "Priority distribution:" << endl;
        cout << "  Low: " << priorityCount[Priority::LOW] << endl;
        cout << "  Medium: " << priorityCount[Priority::MEDIUM] << endl;
        cout << "  High: " << priorityCount[Priority::HIGH] << endl;
        cout << "  Urgent: " << priorityCount[Priority::URGENT] << endl;
        
        cout << "Categories: ";
        for (const auto& cat : categories) {
            cout << cat << " ";
        }
        cout << endl;
        
        auto overdueTasks = getOverdueTasks();
        cout << "Overdue tasks: " << overdueTasks.size() << endl;
    }
};

// TODO: Implement utility functions
string getCurrentTimeString() {
    auto now = system_clock::now();
    time_t time = system_clock::to_time_t(now);
    return string(ctime(&time));
}

// TODO: Implement date parsing function
system_clock::time_point parseDate(const string& dateStr) {
    // Simple implementation - assumes format YYYY-MM-DD HH:MM
    int year, month, day, hour, minute;
    sscanf(dateStr.c_str(), "%d-%d-%d %d:%d", &year, &month, &day, &hour, &minute);
    
    tm timeinfo = {};
    timeinfo.tm_year = year - 1900;
    timeinfo.tm_mon = month - 1;
    timeinfo.tm_mday = day;
    timeinfo.tm_hour = hour;
    timeinfo.tm_min = minute;
    
    time_t time = mktime(&timeinfo);
    return system_clock::from_time_t(time);
}

int main() {
    cout << "=== Task Management System ===" << endl;
    
    TaskManager manager;
    
    // TODO: Add sample tasks
    cout << "\nAdding sample tasks..." << endl;
    
    auto tomorrow = system_clock::now() + hours(24);
    auto nextWeek = system_clock::now() + hours(24*7);
    auto pastDue = system_clock::now() - hours(24);
    
    int task1 = manager.addTask("Complete project proposal", 
                               "Write and submit the quarterly project proposal", 
                               Priority::HIGH, "Work", tomorrow);
    cout << "Added task with ID: " << task1 << endl;
    
    int task2 = manager.addTask("Buy groceries", 
                               "Milk, bread, eggs, fruits", 
                               Priority::MEDIUM, "Personal", nextWeek);
    cout << "Added task with ID: " << task2 << endl;
    
    int task3 = manager.addTask("Fix critical bug", 
                               "Resolve the login authentication issue", 
                               Priority::URGENT, "Work", pastDue);
    cout << "Added task with ID: " << task3 << endl;
    
    int task4 = manager.addTask("Schedule meeting", 
                               "Arrange team sync for next sprint", 
                               Priority::LOW, "Work", tomorrow);
    cout << "Added task with ID: " << task4 << endl;
    
    // TODO: Display all tasks
    manager.displayAllTasks();
    
    // TODO: Update task status
    cout << "\nUpdating task " << task1 << " status to IN_PROGRESS..." << endl;
    manager.updateTaskStatus(task1, TaskStatus::IN_PROGRESS);
    
    // TODO: Display tasks by status
    manager.displayTasksByStatus(TaskStatus::IN_PROGRESS);
    
    // TODO: Find overdue tasks
    cout << "\n=== Overdue Tasks ===" << endl;
    auto overdueTasks = manager.getOverdueTasks();
    if (overdueTasks.empty()) {
        cout << "No overdue tasks." << endl;
    } else {
        for (auto task : overdueTasks) {
            task->display();
        }
    }
    
    // TODO: Search for tasks
    cout << "\n=== Searching for 'project' ===" << endl;
    auto searchResults = manager.searchTasks("project");
    for (auto task : searchResults) {
        task->display();
    }
    
    // TODO: Display tasks by category
    cout << "\n=== Tasks in 'Work' category ===" << endl;
    auto workTasks = manager.getTasksByCategory("Work");
    for (auto task : workTasks) {
        task->display();
    }
    
    // TODO: Show statistics
    manager.getStatistics();
    
    // TODO: Save tasks to file
    try {
        manager.saveToFile("tasks.csv");
    } catch (const exception& e) {
        cout << "Error saving tasks: " << e.what() << endl;
    }
    
    // TODO: Load tasks from file (create a new manager to test loading)
    cout << "\n=== Testing file loading ===" << endl;
    TaskManager newManager;
    try {
        newManager.loadFromFile("tasks.csv");
        newManager.displayAllTasks();
    } catch (const exception& e) {
        cout << "Error loading tasks: " << e.what() << endl;
    }
    
    cout << "\nTask Management System completed successfully!" << endl;
    
    return 0;
}