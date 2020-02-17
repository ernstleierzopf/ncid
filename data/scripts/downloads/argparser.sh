#!/bin/bash

HELP="TODO"
while (( "$#" )); do
  case "$1" in
	-h|--help)
	  echo $HELP
	  echo "Other arguments are ignored with -h or --help!"
	  exit 0
	  ;;
    -e|--extended_download)
      EXT_DOWNLOAD=true
      shift 1
      ;;
	-f|--file-size
	  FS=$2
	  shift 2
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
	  echo "Error: Unsupported argument $1" >&2
      exit 1
      ;;
  esac
done