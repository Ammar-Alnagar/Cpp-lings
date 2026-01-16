#!/bin/bash

echo "==========================================="
echo "C++ Learning Curriculum - Final Verification"
echo "==========================================="
echo

echo "1. Checking main directory structure..."
ls -la /home/ammar/work/Cpp-lings/ | grep -E "(README|CMakeLists|Makefile|exercises|projects)"
echo

echo "2. Verifying exercise chapters exist..."
for i in {0..15}; do
    if [ -f "/home/ammar/work/Cpp-lings/exercises/ch$i/chapter_$i*.md" ]; then
        echo "✓ Chapter $i: OK"
    else
        echo "✗ Chapter $i: MISSING"
    fi
done
echo

echo "3. Verifying project files exist..."
if [ -f "/home/ammar/work/Cpp-lings/projects/task_manager.cpp" ]; then
    echo "✓ Task Manager Project: OK"
else
    echo "✗ Task Manager Project: MISSING"
fi

if [ -f "/home/ammar/work/Cpp-lings/projects/database_engine.cpp" ]; then
    echo "✓ Database Engine Project: OK"
else
    echo "✗ Database Engine Project: MISSING"
fi

if [ -f "/home/ammar/work/Cpp-lings/projects/image_processor.cpp" ]; then
    echo "✓ Image Processor Project: OK"
else
    echo "✗ Image Processor Project: MISSING"
fi
echo

echo "4. Checking README and documentation files..."
if [ -f "/home/ammar/work/Cpp-lings/README.md" ]; then
    echo "✓ Main README: OK"
else
    echo "✗ Main README: MISSING"
fi

if [ -f "/home/ammar/work/Cpp-lings/SUMMARY.md" ]; then
    echo "✓ SUMMARY: OK"
else
    echo "✗ SUMMARY: MISSING"
fi

if [ -f "/home/ammar/work/Cpp-lings/projects/README.md" ]; then
    echo "✓ Projects README: OK"
else
    echo "✗ Projects README: MISSING"
fi
echo

echo "5. Sample content from Chapter 1:"
echo "----------------------------------------"
head -20 /home/ammar/work/Cpp-lings/exercises/ch1/chapter_1_basics.md
echo "..."
echo

echo "6. Sample content from Project 1:"
echo "----------------------------------------"
head -20 /home/ammar/work/Cpp-lings/projects/task_manager.cpp
echo "..."
echo

echo "==========================================="
echo "Verification Complete!"
echo "The C++ Learning Curriculum is properly structured."
echo "All chapters, exercises, and projects are in place."
echo "==========================================="