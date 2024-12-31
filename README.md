Introduction

The "Customer Queues Simulation" project explores different queuing systems using Python, focusing on the application of queuing theory to model customer wait times and service processes. The project is organized into four main directories, each dedicated to different aspects of queuing, including customer priority handling, multiple server systems, simple queues, and variance reduction techniques. Through these simulations, the project aims to provide a deeper understanding of queuing dynamics and the impact of various policies and configurations on system performance.

Directory Structure

The project is structured into four directories, each containing simulations and associated reports on specific queuing models. The Simple_Queue directory compares simple queuing systems under FIFO (First-In, First-Out) and LIFO (Last-In, First-Out) disciplines, with files implementing the M/M/1 queue under both rules. The Multiple_Servers directory explores the performance of multiple server systems, including shared and tandem queues, with simulations analyzing metrics such as mean delay and system performance. The Customers_with_priority directory focuses on queuing systems with customer priority rules, simulating scenarios with and without preemption. Finally, the Variance_Reduction directory demonstrates techniques for reducing variance in simulations, applying methods like antithetic variates and control variates to improve accuracy.

Reports

Each directory includes a PDF report that explains the simulation methods, experimental setup, and results for the respective queuing systems. These reports provide insights into various scenarios explored, such as the impact of customer priority and multiple servers on system performance, and the application of variance reduction techniques. They also include analysis of key metrics, such as mean delays and system utilization, and draw conclusions about the effectiveness of different queuing strategies in real-world applications.

How to Run the Notebooks

To run the simulations, clone the repository and navigate to the desired directory. Once inside the directory, you can execute the Python files or Jupyter notebooks to explore the respective simulations. The instructions provided in the repository allow for easy execution of the notebooks:

git clone https://github.com/saurabhshivpuje/Customer_Queues.git
cd Customer_Queues
cd <directory_name>
python3 <python_file_name> 
or 
jupyter <notebook_name>
Summary

This project provides an in-depth exploration of various queuing models and simulation techniques. It covers customer priority handling, multiple server systems, and methods for improving simulation accuracy through variance reduction. The simulations and corresponding reports offer valuable insights into queuing theory concepts and their practical applications, making this project a useful resource for understanding and optimizing customer queuing systems.
