for i in 0 1
do
printf "n_free node$i 2M: "
cat /sys/devices/system/node/node$i/hugepages/hugepages-2048kB/free_hugepages
printf "n_free node$i 1G: "
cat /sys/devices/system/node/node$i/hugepages/hugepages-1048576kB/free_hugepages
done
cat /proc/meminfo | grep 'AnonHugePages'
