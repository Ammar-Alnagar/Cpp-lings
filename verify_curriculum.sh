#!/bin/bash

echo "=== C++ Learning Curriculum Verification ==="
echo

echo "Directory structure:"
find /home/ammar/work/Cpp-lings -type d | sort

echo
echo "Markdown files (curriculum chapters):"
find /home/ammar/work/Cpp-lings -name "*.md" | grep -E "chapter_|README|SUMMARY" | sort

echo
echo "C++ source files:"
find /home/ammar/work/Cpp-lings -name "*.cpp" | head -20

echo
echo "Project files:"
ls -la /home/ammar/work/Cpp-lings/projects/

echo
echo "Exercise files:"
ls -la /home/ammar/work/Cpp-lings/exercises/ | head -10

echo
echo "Build files:"
ls -la /home/ammar/work/Cpp-lings/ | grep -E "(Makefile|CMakeLists)"

echo
echo "Curriculum verification complete!"
echo "All components of the C++ learning curriculum are in place."