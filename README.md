# AU-Project
The compressed file contains all Folders in the repository, with additional files too large for upload.
The "Model" folder contains AMPL model files representing the optimization problem.
The "Data" folder has the input files extracted from Linksbridge and other sources, plus the Python file "create_data.py". This is the file that reads and formats the input data, then runs the AMPL model to generate the files "var.csv" and "summary.csv" in the DataIn sub-folder.
The "Extended Analysis" folder contains the "create_graph.py" script, which uses the output of "create_data.py" to generate multiple groups of graphs. Graphs that study the effect of a particular factor have "-effect" as a suffix. Graphs that focus on a single market are in the "Market" sub-folder. The remaining files compare the different markets when using different MPRs or Negotiation Order.
The "Presentation" folder contains the "AU_Overview.odp" file, which offers an overview of the how the data was prepared, the factors being tested, and the main results.
