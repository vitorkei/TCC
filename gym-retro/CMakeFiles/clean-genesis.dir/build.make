# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kei/gym-retro

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kei/gym-retro

# Utility rule file for clean-genesis.

# Include the progress variables for this target.
include CMakeFiles/clean-genesis.dir/progress.make

CMakeFiles/clean-genesis:
	cd /home/kei/gym-retro/cores/genesis && $(MAKE) -f Makefile.libretro clean
	cd /home/kei/gym-retro/cores/genesis && /usr/local/bin/cmake -E remove /home/kei/gym-retro/retro/cores/genesis_plus_gx_libretro.so

clean-genesis: CMakeFiles/clean-genesis
clean-genesis: CMakeFiles/clean-genesis.dir/build.make

.PHONY : clean-genesis

# Rule to build all files generated by this target.
CMakeFiles/clean-genesis.dir/build: clean-genesis

.PHONY : CMakeFiles/clean-genesis.dir/build

CMakeFiles/clean-genesis.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/clean-genesis.dir/cmake_clean.cmake
.PHONY : CMakeFiles/clean-genesis.dir/clean

CMakeFiles/clean-genesis.dir/depend:
	cd /home/kei/gym-retro && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kei/gym-retro /home/kei/gym-retro /home/kei/gym-retro /home/kei/gym-retro /home/kei/gym-retro/CMakeFiles/clean-genesis.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/clean-genesis.dir/depend

