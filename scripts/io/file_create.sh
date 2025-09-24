#!/bin/bash

# get this shell script's directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

g++ -O3 -o "${current_dir}/file_generator" "${current_dir}/file_generator.cpp"

save_dir="${current_dir}/../../data/synthetic/transactional"
# Create the synthetic directory if it does not exist
mkdir -p $save_dir

# Check if compilation was successful
if [[ $? -ne 0 ]]; then
    echo "Compilation failed."
    exit 1
fi

file_gen_loc="${current_dir}/file_generator"


# sizes=("1024M" "2048M" "4096M")

sizes=("8192M" "16384M")

shapes=("square")
# shapes=("triangle")

delimiter=","

# Generate files
for size in "${sizes[@]}"; do
    for shape in "${shapes[@]}"; do
        fileName="${shape}_${size}.csv"

        # Generate the file
        fileName="${save_dir}/${fileName}"

        echo "Generating ${fileName}..."
        if [[ "$shape" == "square" ]]; then
            echo "$fileName" | $file_gen_loc $fileName $size $delimiter 1 > /dev/null
        elif [[ "$shape" == "triangle" ]]; then
            echo "$fileName" | $file_gen_loc $fileName $size $delimiter 2 > /dev/null
        fi
    done
done

echo "File generation complete."