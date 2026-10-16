#!/usr/bin/env bash

for pdf in *.pdf; do
    echo $pdf;
    pdftoppm -jpeg -r 300 -singlefile $pdf ${pdf%.*};
done
