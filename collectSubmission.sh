rm -f assignment2.zip 
zip -r assignment2.zip . -i "Problem_*.ipynb" "lib/*.py" "lib/tf_models/*.py" -x "*.ipynb_checkpoints*"
