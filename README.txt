Authors:
	Elad Pezarker
    


requirements:
    Python Version : 3.8.6
    all project requirements located at requirements.txt file

    to install all requirements run the following command:
        pip install -r requirements.txt


run program:
    cd to python script directory

    General command line usage:
        -- To Run The Scanner on entire folder use :
                            'python scanner.py -f <input-folder-path>'
                             script will load all '*.jpg' images and run the scanner on them
                             output will be saved in 'output' folder in input folder path
                      OR:
                            To scan a single file use:
                            'python scanner.py <input-img-path> <output-img-path>'
                            script will load the input image and save the output
                            to given output path.



                      Examples:
                            'python scanner.py -f ./inputs/'
                            'python scanner.py ./inputs/Game.jpg ./output.jpg'



## Project was made on Windows 10 -- OS Build 19042.746


