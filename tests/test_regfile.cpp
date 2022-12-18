#include <gtest/gtest.h>
#include "reg_file.hpp"

TEST(reg, reg_file)
{
    Sim::RegisterFile reg_file;

    unsigned thread_num0 = 1 << 30;
    unsigned thread_num1 = 1 << 20;
    unsigned value0 = 1234;
    unsigned value1 = 5678;

    reg_file.write(thread_num0, 0, value0);
    reg_file.write(thread_num1, 0, value1);
    EXPECT_EQ(reg_file.read(thread_num0, 0), value0);
    EXPECT_EQ(reg_file.read(thread_num1, 0), value1);

    reg_file.write(thread_num0, 255, value0);
    EXPECT_EQ(reg_file.read(thread_num0, 255), 0);

    reg_file.reset();
    EXPECT_EQ(reg_file.read(thread_num0, 0), 0);
}

TEST(reg, preg_file)
{
    Sim::PRegisterFile preg_file;

    unsigned thread_num0 = 1 << 30;
    unsigned thread_num1 = 1 << 20;
    bool value0 = true;
    bool value1 = false;

    preg_file.write(thread_num0, 0, value0);
    preg_file.write(thread_num1, 0, value1);
    EXPECT_EQ(preg_file.read(thread_num0, 0), value0);
    EXPECT_EQ(preg_file.read(thread_num1, 0), value1);

    preg_file.write(thread_num0, 7, value0);
    EXPECT_EQ(preg_file.read(thread_num0, 7), true);

    preg_file.reset();
    EXPECT_EQ(preg_file.read(thread_num0, 0), false);
}
