#!/bin/bash

# Test assignment functionality by running individual commands

echo "Testing immediate assignment x = 42"
echo "x = 42" > /tmp/test1.lyra
cargo run --bin lyra run /tmp/test1.lyra

echo -e "\nTesting symbol retrieval x"
echo "x" > /tmp/test2.lyra
cargo run --bin lyra run /tmp/test2.lyra

echo -e "\nTesting computation y = x + 8"
echo "y = x + 8" > /tmp/test3.lyra
cargo run --bin lyra run /tmp/test3.lyra

echo -e "\nTesting symbol retrieval y"
echo "y" > /tmp/test4.lyra
cargo run --bin lyra run /tmp/test4.lyra