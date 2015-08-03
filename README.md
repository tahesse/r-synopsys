# r-ml-synopsys
Synopsys of machine learning, robot learning and robotics etc.


# documentation:

## folder structure:
- every directory prefixed with a number smaller than 99 contains a media folder
  - (corresponding images should be put in there)
- the 99-style directory contains all .sty files for this project

## how to add a new chapter?
- in the /synopsys.tex include your file(s) using the directory variables and
  add your includes to the includelist
  
## what commands to use?
- \fixcfigure[!]{0.0-1.0}{\path/\to/image.png}  creates a fixed centered figure
  argument 1: optional (default: ! keep the scale of the image fixed) sets the 
              height of the image
  argument 2: sets the width of the image in percent, i.e. arg2 \in {0.0 - 1.0}
  argument 3: absolute or relative path the image, the extension may be omitted
  
- \tocaption{[...]} shall be used after \fixcfigure and adds a caption to it

- \emph to emphasize text

- \vec should be used for vectors and \mat should be used for matrices, both can
  be used in text mode, i.e. no $ $ wrapping the command are required
  
- \itab with its extension \tab, \htab, \dtab can be used to create structured 
  itemize blocks

## conventions:
- when you put a label on something, always prefix with part of its environment:
  e.g. table -> tab:name, chapter -> chap:name, section -> sec:name, 
       subsection -> subsec:name, equation -> eq:name, etc.

- tend to use as many wrappers as often

- chapter section hierarchy is as follows: chapter > section > subsection
                                                   > subsubsection > paragraph

- for notation in formulars, please read the notation chapter that came along 
  with these contents


# LaTeX2e Macros
http://ctan.math.utah.edu/ctan/tex-archive/info/macros2e/macros2e.pdf