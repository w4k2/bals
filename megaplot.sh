#!/bin/bash
cd plots
montage drift-0-0.png drift-1-0.png drift-2-0.png drift-3-0.png drift-4-0.png drift-5-0.png -geometry +2+2  megaplot-0.png
montage drift-0-1.png drift-1-1.png drift-2-1.png drift-3-1.png drift-4-1.png drift-5-1.png -geometry +2+2  megaplot-1.png
