# K-Nearest Neighbors, Neural Networks, and Naive Bayes
Implementation of  K-Nearest Neighbors, Neural Networks, and Naive Bayes from scratch

Data Mining Final Project
----------------------------------------
The project is being delivered in a computational notebook created in the 
Jupyter Notebook application. These instruction assume the user is using the 
Jupyter Notebook application to view the final project notebook. Our program
is created to have the ability to either select from a list of datasets or 
select the file name for a .csv file with a personalized dataset.

------------
STEPS
------------
Step 1: Select the .csv file to utilize as the algorithm's dataset

Step 2: Select the classification value column index

Step 3: Select the classification method to implement on the dataset

Step 4: Output / Comparison

----------------------------------------------------------------------------

Step 1:

	From the toolbar at the top of the windel select "Cell", then select the
	'"Run All" option. The application will print out onto the console a list 
	of options for selecting which dataset's .csv file to implement the algorithm 
	with. The option choices are as follows:

		1 - iris.csv
		2 - wine.csv
		3 - wdbc.csv
		4 - Enter custom filename

	The program will wait for the user to input the number that corresponds 
	to the dataset they wish to select. In order to select a dataset the user
	must enter a number then press the "Enter" key. If the user decides to
	use a personalized dataset, the user will then be prompted to enter the 
	the filename of the .csv file containing the dataset values. 

Step 2:

	Since our program has the ability to use a personalized dataset, it is 
	necessary to specify the column that represents the classification value
	in these cases. If the user selected any of the first three datasets, the 
	program will automatically decide the column depending on which dataset is 
	selected. If the user chose to use a personalized dataset the program will
	at this point ask the user to enter the column index for the classification
	value. (Indexes starting at zero)

Step 3: 

	Now the program will print onto the console a menu of options for selecting 
	which classification algorithm to apply to the dataset. The options are
	as follows: 
	
		1 - KNN (K Nearest Neighbors)
		2 - Bayes(Naive Bayes)
		3 - Neural Nets (Neural Network)

	The program will wait for the user to input the number that corresponds 
	to the algorithm they wish to select. In order to select an algorithm the 
user must enter a number then press the "Enter" key. 

If the user enters the option choice for the K Nearest Neighbors 
algorithm they will be asked to enter a value for k (number of nearest neighbor). 
If the user presses the "Enter" key without inputting a value the default value 
of 3 will be used. 

If the user enters the option choice for the Neural Network algorithm 
they will be asked to enter multiple parameters: learning rate, number of epochs,
number of layers, and number of neurons on each layer. If the user presses the 
"Enter" key without inputting a value the default value of 0.15, 100, 2, 3, and 3 will 
be used for learning rate, number of epochs, number of layers, neurons for first
layer, and number of neurons for the second layer respectively.

Step 4:
	OUTPUT
	=========
	KNN
	The application's output for this algorithm is a list of the accuracy scores 
	obtained from a 10-fold cross-validation, as well as the mean accuracy.    

	Naive
	The application's output for this algorithm is a list of the accuracy scores 
	obtained from a 10-fold cross-validation, as well as the mean accuracy.    

	Neural Net
	The application's output for this algorithm is a list of the accuracy scores 
	obtained from a 10-fold cross-validation, as well as the mean accuracy.    


	COMPARISON
	==========
	The final cell in the notebook will display a list of accuracy scores obtained
	by using the sklearn library's implementation of our chosen classification
	algorithms on each of our selected datasets. This list is printed for the 
	purpose of being able to compare the accuracy metrics of our implementations
	with the implementations from the sklearn library.  		