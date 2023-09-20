#!/bin/bash
#pip install -e ..
#optional parameters
while [[ $# -gt 0 ]]
do
	case "$1" in
	    -s|--server)
	    server="$2"
	    shift # past argument
	    shift # past value
	    ;;
	    -e|--exp_name)
	    expt_name="$2"
	    shift # past argument
	    shift # past value
	    ;;
	    -c|--config_file)
	    config_file="$2"
	    shift # past argument
	    shift # past value
	    ;;
	     -o|--output_path)
	    output_path="$2"
	    shift # past argument
	    shift # past value
	    ;;
	    *)
		shift
		echo "Invalid option -$1" >&2
	    ;;
	esac
done
set -e

server=${server:=local}
dispatcher config --exp_name ${expt_name} --output_path ${output_path} --config_file ${config_file}

echo $server
case ${server} in
  local)
    dispatcher dispatch --exp_name ${expt_name} --output_path ${output_path}
    ;;

  gra)
    sbatch submit.sh ${expt_name} $output_path
   ;;

  *)
    echo "Mode is invalid. Options are debug or prod."
    exit 1
    ;;
esac
