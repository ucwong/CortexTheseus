for x in $(find ./ -name "*.go"); do sed -i  's/\/CortexFoundation\/CortexTheseus/\/ethereum\/go-ethereum/g' $x ;done
