#!/bin/bash

#                           Node 0          Node 1           Total
#                  --------------- --------------- ---------------
# Numa_Hit                85806.66       637699.93       723506.59
# Numa_Miss               72411.09         7119.75        79530.84
# Numa_Foreign             7119.75        72411.00        79530.75
# Interleave_Hit              2.48            2.41            4.88
# Local_Node              85565.79       637683.78       723249.56
# Other_Node              72651.96         7135.91        79787.86

last_hit=(0 0)
last_miss=(0 0)
last_foreign=(0 0)
last_interleave=(0 0)
last_local=(0 0)
last_other=(0 0)

interval=2

while true; do
	clear
	date
	stats=`numastat -n`
	for node in 0 1; do
		col=$(( node + 2 ))
		for row in "Numa_Hit" "Numa_Miss" "Numa_Foreign" "Interleave_Hit" "Local_Node" "Other_Node"; do
			val=`echo "$stats" | grep $row | awk '{print $'"$col"'}'`
			if [ $row == "Numa_Hit" ]; then
				delta=`echo "$val - ${last_hit[$node]}" | bc`
				last_hit[$node]=$val
			elif [ "$row" == "Numa_Miss" ]; then
				delta=`echo "$val - ${last_miss[$node]}" | bc`
				last_miss[$node]=$val
			elif [ "$row" == "Numa_Foreign" ]; then
				delta=`echo "$val - ${last_foreign[$node]}" | bc`
				last_foreign[$node]=$val
			elif [ "$row" == "Interleave_Hit" ]; then
				delta=`echo "$val - ${last_interleave[$node]}" | bc`
				last_interleave[$node]=$val
			elif [ "$row" == "Local_Node" ]; then
				delta=`echo "$val - ${last_local[$node]}" | bc`
				last_local[$node]=$val
			elif [ "$row" == "Other_Node" ]; then
				delta=`echo "$val - ${last_other[$node]}" | bc`
				last_other[$node]=$val
			else
				continue
			fi
			printf "node$node $row:\t"
			if [ "$row" == "Numa_Hit" ]; then
				printf "\t"
			fi
			echo "$delta / $interval" | bc
		done
		echo "-----------"
	done
	echo "-------------------------"
	sleep $interval
done
