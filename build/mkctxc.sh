for x in $(find ./ -name "*.go"); do sed -i -e 's/\/ethereum\/go-ethereum/\/CortexFoundation\/CortexTheseus/g' $x ;done
