cmake_minimum_required(VERSION 3.8)
project(CppExercises)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Function to create an executable for each exercise file
function(add_exercise_executable subdir filename)
    get_filename_component(exercise_name ${filename} NAME_WE)
    set(target_name "${subdir}_${exercise_name}")
    add_executable(${target_name} ${subdir}/${filename})
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/exercises/${subdir}
    )
endfunction()

# Chapter 0 exercises
file(GLOB CH0_EXERCISES "ch0/*.cpp")
foreach(exercise_file ${CH0_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch0" ${exercise_name})
endforeach()

# Chapter 1 exercises
file(GLOB CH1_EXERCISES "ch1/*.cpp")
foreach(exercise_file ${CH1_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch1" ${exercise_name})
endforeach()

# Chapter 2 exercises
file(GLOB CH2_EXERCISES "ch2/*.cpp")
foreach(exercise_file ${CH2_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch2" ${exercise_name})
endforeach()

# Chapter 3 exercises
file(GLOB CH3_EXERCISES "ch3/*.cpp")
foreach(exercise_file ${CH3_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch3" ${exercise_name})
endforeach()

# Chapter 4 exercises
file(GLOB CH4_EXERCISES "ch4/*.cpp")
foreach(exercise_file ${CH4_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch4" ${exercise_name})
endforeach()

# Chapter 5 exercises
file(GLOB CH5_EXERCISES "ch5/*.cpp")
foreach(exercise_file ${CH5_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch5" ${exercise_name})
endforeach()

# Chapter 6 exercises
file(GLOB CH6_EXERCISES "ch6/*.cpp")
foreach(exercise_file ${CH6_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch6" ${exercise_name})
endforeach()

# Chapter 7 exercises
file(GLOB CH7_EXERCISES "ch7/*.cpp")
foreach(exercise_file ${CH7_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch7" ${exercise_name})
endforeach()

# Chapter 8 exercises
file(GLOB CH8_EXERCISES "ch8/*.cpp")
foreach(exercise_file ${CH8_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch8" ${exercise_name})
endforeach()

# Chapter 9 exercises
file(GLOB CH9_EXERCISES "ch9/*.cpp")
foreach(exercise_file ${CH9_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch9" ${exercise_name})
endforeach()

# Chapter 10 exercises
file(GLOB CH10_EXERCISES "ch10/*.cpp")
foreach(exercise_file ${CH10_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch10" ${exercise_name})
endforeach()

# Chapter 11 exercises
file(GLOB CH11_EXERCISES "ch11/*.cpp")
foreach(exercise_file ${CH11_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch11" ${exercise_name})
endforeach()

# Chapter 12 exercises
file(GLOB CH12_EXERCISES "ch12/*.cpp")
foreach(exercise_file ${CH12_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch12" ${exercise_name})
endforeach()

# Chapter 13 exercises
file(GLOB CH13_EXERCISES "ch13/*.cpp")
foreach(exercise_file ${CH13_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch13" ${exercise_name})
endforeach()

# Chapter 14 exercises
file(GLOB CH14_EXERCISES "ch14/*.cpp")
foreach(exercise_file ${CH14_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch14" ${exercise_name})
endforeach()

# Chapter 15 exercises
file(GLOB CH15_EXERCISES "ch15/*.cpp")
foreach(exercise_file ${CH15_EXERCISES})
    get_filename_component(exercise_name ${exercise_file} NAME)
    add_exercise_executable("ch15" ${exercise_name})
endforeach()

message(STATUS "C++ Exercises CMake project configured")
message(STATUS "Use 'cmake --build .' to build all exercises")