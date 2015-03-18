/**
 * This class models a basic multilayer perceptron with a variable amount of 
 * hidden nodes, 2 input nodes, and 1 output node. The perceptron attempts
 * to model the XOR function.
 * 
 * @author Hemant Kunda
 * @date September 21, 2014
 */
public class MultilayerNetwork {
   // Final values of the network 
   static final int NUM_INPUTS = 2;
   static final int NUM_HIDDEN = 5;
   static final int NUM_OUTPUTS = 2;
   static final int NUM_WEIGHTS_KJ = NUM_INPUTS*NUM_HIDDEN;
   static final int NUM_WEIGHTS_JI = NUM_HIDDEN*NUM_OUTPUTS;
   static final int NUM_MODELS = (int)(Math.pow(2, NUM_INPUTS));
   static final double LAMBDA = 1;
   //Limit on # of back propagations
   static final int RECALC_THRESHOLD = 1000000;
   //Largest acceptable error value
   static final double ERROR_THRESHOLD = .0001;
   
   // Limits on initial weights
   static final double MAX_INITIAL_WEIGHT = 5;
   static final double MIN_INITIAL_WEIGHT = -5;
   
   // Storage of the layers and weights
   private int[][] inputNodes = new int[NUM_INPUTS][NUM_MODELS]; 
   private double[] hiddenLayer = new double[NUM_HIDDEN];
   private double[][] weights_ji = new double[NUM_HIDDEN][NUM_OUTPUTS];
   private double[][] weights_kj = new double[NUM_INPUTS][NUM_HIDDEN];
   private double[][] output = new double[NUM_OUTPUTS][NUM_MODELS];
   private int[][] trainingSet = new int[NUM_OUTPUTS][NUM_MODELS];
   
   public MultilayerNetwork()
   {
      
   }
   
   /*
    * Initializes the weights with random values, prints out the initial information for 
    * the perceptron, converges the error function, and prints out the final data.
    */
   public static void main(String[] args)
   {
      MultilayerNetwork network = new MultilayerNetwork();
      network.runNetwork();
   }
   
   /*
    * Instantiates the inputs and weights, calculates the initial outputs, 
    * prints out the initial information, adjusts the weights based off the gradients, 
    * and prints out the final (converged) information.
    */
   public void runNetwork()
   {
      initializeInputs(0, NUM_INPUTS - 1, 0, NUM_MODELS - 1);
      createXORTrainingSet();
      createXNORTrainingSet();
      randomizeWeights();
      calculateInitialOutput();
      printInitialInfo();
      //adjustWeights();
      backPropagate();
      printFinalInfo();
   }
   
   /*
    * Initializes each weight with a random value between MAX_INITIAL_WEIGHT and
    * MIN_INITIAL_WEIGHT
    */
   public void randomizeWeights()
   {
      for (int k = 0; k < NUM_INPUTS; k++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            weights_kj[k][j] = (Math.random() * MAX_INITIAL_WEIGHT + 
                                    Math.random() * MIN_INITIAL_WEIGHT);
         }
      }
      
      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            weights_ji[j][i] = (Math.random() * MAX_INITIAL_WEIGHT + 
                                    Math.random() * MIN_INITIAL_WEIGHT);
         }
      }
   }
   
   /*
    * Creates the XOR training set in the first layer of the training set variable.
    * The output of XOR with multiple outputs is determined by the number of 1's for 
    * several reasons:
    *    1) All 0's are insignificant. This is because 0 XOR 1 = 1, and 0 XOR 0 = 0,
    *    suggesting that the 0 doesn't do anything to change the output. Thus, we ignore
    *    all 0's.
    *    2) 1 XOR 1 = 0, which we determined we'd ignore. Thus, we ignore all pairs of 1.
    * 
    * By ignoring all 0's and pairs of 1, the output of XOR utimately depends on whether
    * the total number of 1's is even or odd.  An odd number of 1's corresponds to an 
    * output of 1, while an even number of 1's corresponds to an output of 0.
    */
   public void createXORTrainingSet()
   {
      for (int m = 0; m < NUM_MODELS; m++)
      {
         int trainingVal = 0;
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            trainingVal += inputNodes[k][m];
         }
         trainingVal = trainingVal % 2;
         trainingSet[0][m] = trainingVal;
      }
   }
   
   /*
    * Creates the XNOR training set in the second layer of the training set variable.
    * As XNOR = !XOR, the training set is created using the values of the XOR that
    * was previously generated.  
    */
   public void createXNORTrainingSet()
   {
      for (int m = 0; m < NUM_MODELS; m++)
      {
         trainingSet[1][m] = 1 - trainingSet[0][m];
      }
   }
   
   /*
    * Recursively implements the input variable with all possible combinations of 
    * 0 and 1, starting from the startR row to the endR row.
    * 
    * @param startR the beginning row
    * @param endR the end row (included)
    * @param startC the starting column
    * @param endC the end column (included)
    * 
    * Precondition: startR > 0, startC > 0, endR < NUM_INPUTS, endC < NUM_MODELS
    */
   public void initializeInputs(int startR, int endR, int startC, int endC)
   {
      if (startR > endR)
      {
         return;
      }
      int midC = (startC + endC) / 2;
      for (int c = startC; c <= midC; c++)
      {
         inputNodes[startR][c] = 1;
      }
      for (int c = midC + 1; c <= endC; c++)
      {
         inputNodes[startR][c] = 0;
      }
      initializeInputs(startR + 1, endR, startC, midC);
      initializeInputs(startR + 1, endR, midC + 1, endC);
   }
   
   /*
    * Prints out the initial weights, initial outputs and initial error.
    */
   public void printInitialInfo()
   {
      System.out.println("Initial Weights: \n");
      System.out.println("Weights_kj: \n");
      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            System.out.println("\tw" + (k+1) + (j+1) + ": " + weights_kj[k][j]);
         }
      }
      
      System.out.println("\nWeights_ji: \n");
      for (int i = 0; i < NUM_OUTPUTS; i++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            System.out.println("\tw" + (j+1) + (i+1) + ": " + weights_ji[j][i]);
         }
      }
      
      System.out.println("\nInitial Outputs: ");
      for (int m = 0; m < NUM_MODELS; m++)
      {
         System.out.println("Model " + (m+1) + ": ");
         System.out.println("Expected: ");
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            System.out.println("\tF" + (i+1) + ": " + trainingSet[i][m]);
         }
         System.out.println();
         System.out.println("Actual: ");
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            System.out.println("\tF" + (i+1) + ": " + output[i][m]);
         }
         System.out.println();
      }
      System.out.println("\nInitial Error: " + errorFunction());
   }
   
   /*
    * Prints out the final weights, final outputs and final (converged) error.
    */
   public void printFinalInfo()
   {
      System.out.println("Final Weights: \n");
      System.out.println("Weights_kj: \n");
      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            System.out.println("\tw" + (k+1) + (j+1) + ": " + weights_kj[k][j]);
         }
      }
      
      System.out.println("\nWeights_ji: \n");
      for (int i = 0; i < NUM_OUTPUTS; i++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            System.out.println("\tw" + (j+1) + (i+1) + ": " + weights_ji[j][i]);
         }
      }
      System.out.println("\nFinal Outputs: ");
      for (int m = 0; m < NUM_MODELS; m++)
      {
         System.out.println("Model " + (m+1) + ": ");
         System.out.println("Expected: ");
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            System.out.println("\tF" + (i+1) + ": " + trainingSet[i][m]);
         }
         System.out.println();
         System.out.println("Actual: ");
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            System.out.println("\tF" + (i+1) + ": " + output[i][m]);
         }
         System.out.println();
      }
      System.out.println("\nFinal Error: " + errorFunction());
   }
   
   /*
    * Continually runs the back propagation until the error falls below the
    * error threshold or the back propagation has run more times than the
    * recalculation threshold.
    */
   public void backPropagate()
   {
      int i = 0;
      int m = 0;
      while (errorFunction() > ERROR_THRESHOLD && i < RECALC_THRESHOLD)
      {
         calculateOutput(m);
         singleBackPropagation(m);
         i++;
         m++;
         if (m >= NUM_MODELS)
         {
            m = 0;
         }
      }
      System.out.println("\nAfter " + i + " iterations: \n");
   }
   /*
    * Loops through each model, calculating the change in each weight in the kj layer
    * by first calculating the change in each weight in the ji layer.  By doing so, one
    * of the variables needed for the change in the weight in the kj layer is also 
    * calculated, saving run-time resources.
    */
   public void singleBackPropagation(int m)
   {
      for (int k = 0; k < NUM_INPUTS; k++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            double Omega_j = 0;
            for (int i = 0; i < NUM_OUTPUTS; i++)
            {
               double psi_i = psi_i(i, m);
               double deltaWeights_ji = LAMBDA * hiddenLayer[j] * psi_i;
               weights_ji[j][i] = weights_ji[j][i] + deltaWeights_ji;
               Omega_j += psi_i * weights_ji[j][i];      

            }
            double Theta_j = Theta_j(j, m);
            double Psi_j = Psi_j(Omega_j, Theta_j);
            double deltaWeights_kj = LAMBDA * inputNodes[k][m] * Psi_j;
            weights_kj[k][j] = weights_kj[k][j] + deltaWeights_kj;
         }
      }
   }
   
   public void calculateInitialOutput()
   {
      for (int m = 0; m < NUM_MODELS; m++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            double sum = 0;
            for (int k = 0; k < NUM_INPUTS; k++)
            {
               sum += inputNodes[k][m]*weights_kj[k][j];
            }
            hiddenLayer[j] = sigmoid(sum);
         }
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            double sum = 0;
            for (int j = 0; j < NUM_HIDDEN; j++)
            {
               sum += hiddenLayer[j]*weights_ji[j][i];
            }
            output[i][m] = sigmoid(sum);
         }
      }
   }
   /*
    * Calculates the values of the hidden layer, then calculates the values of the
    * output.
    */
   public void calculateOutput(int m)
   {
      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         double sum = 0;
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            sum += inputNodes[k][m]*weights_kj[k][j];
         }
         hiddenLayer[j] = sigmoid(sum);
      }
      for (int i = 0; i < NUM_OUTPUTS; i++)
      {
         double sum = 0;
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            sum += hiddenLayer[j]*weights_ji[j][i];
         }
         output[i][m] = sigmoid(sum);
      }
   }
   
   /*
    * Loops over each model, calculating the value of each hidden layer node
    * by summing the product of each weight connected to the node and the input
    * value attached to that weight and passing that value through the sigmoid
    * function. 
    */
   public void sumIntoHidden()
   {
      
   }
   /*
    * Loops over each model, calculating the value of each output node by summing
    * the product of each weight connected to the output node and the value of the
    * hidden node attached to that weight and passing that value through the 
    * sigmoid function.
    */
   public void sumIntoOutput()
   {
      for (int m = 0; m < NUM_MODELS; m++)
      {
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            double sum = 0;
            for (int j = 0; j < NUM_HIDDEN; j++)
            {
               sum += hiddenLayer[j]*weights_ji[j][i];
            }
            output[i][m] = sigmoid(sum);
         }
      }
   }
   
   /*
    * Calculates the error function.  The error function is half of
    * the square of the output values subtracted from the training values, summed 
    * over all the models.
    * 
    * @return the value of the error function
    */
   public double errorFunction()
   {
      double sum = 0;
      for (int m = 0; m < NUM_MODELS; m++)
      {
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            sum += (trainingSet[i][m] - output[i][m]) * (trainingSet[i][m] - output[i][m]);
         }
      }
      return .5 * sum;
   }
   
   /*
    * Calculates omega_i, which is equal to the difference between the training value
    * and output value, each of which is defined by i (location in the output layer) and
    * m (model number).
    * 
    * @param i the index of the output node to consider
    * @param m the model to consider
    * @return the value of omega_i
    */
   public double omega_i(int i, int m)
   {
      return (trainingSet[i][m] - output[i][m]);
   }
   
   /*
    * Calculates psi_i, which is equal to the product of omega_i and Theta_i.
    * 
    * @param i the index of the output node to consider
    * @param m the model to consider
    * 
    * @return the value of psi_i
    */
   public double psi_i(int i, int m)
   {
      return omega_i(i, m) * sigmoidDeriv(Theta_i(i, m));
   }
   
   /*
    * Calculates psi_i, which is equal to the product of omega_i and Theta_i.
    * 
    * This version of the method obviates calculating omega_i and Theta_i in
    * the event that those values have already been stored.
    * 
    * @param omega_i the value of omega_i, the sum over the models
    * of the difference between the training value and output value
    * @param Theta_i the value of Theta_i, the sum over the models of the 
    * sum over the hidden layer nodes of the product of a hidden node and its
    * corresponding weight
    * 
    * @return the value of psi_i
    */
   public double psi_i(double omega_i, double Theta_i)
   {
      return omega_i * sigmoidDeriv(Theta_i);
   }
  
   /*
    * Calculates Theta_i, which is equal to the sum over the hidden layer nodes of 
    * the product of each hidden node and its corresponding weight.
    * 
    * @param i the index of the output node to consider
    * @param m the model to consider
    * 
    * @return the value of Theta_i
    */
   public double Theta_i(int i, int m)
   {
      double Theta = 0;

      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         Theta += hiddenLayer[j] * weights_ji[j][i];
      }
      return Theta;
   }
   
   /*
    * Calculates Psi_j, which is equal to the product of Omega_j and the 
    * derivative of the sigmoid of Theta_j.  
    * 
    * This version of the method obviates calculating Omega_j and Theta_j in
    * the event that those two have already been calculated.
    * 
    * @param Omega_j the value of Omega_j,
    */
   public double Psi_j(double Omega_j, double Theta_j)
   {
      return Omega_j * sigmoidDeriv(Theta_j);
   }

   /*
    * Calculates the value of Theta_j, which is equal to the sum over the inputs of 
    * the product of each input and the weight attaching it to the specified hidden
    * node.
    * 
    * @param j the index hidden node to consider
    * @param m the model to consider
    * 
    * @return the value of Theta_j
    */
   public double Theta_j(int j, int m)
   {
      double Theta_j = 0;
      for (int k = 0; k < NUM_INPUTS; k++)
      {
         Theta_j += inputNodes[k][m] * weights_kj[k][j];
      }
      return Theta_j;
   }
   /*
    * Calculates the gradient for each of the weights between the k and j layers and 
    * returns them in a 2D array.  
    * 
    * The gradient for w_kj is calculated through the product of 6 numbers:
    * 
    * a) LAMBDA, the reduction constant
    * b) the derivative of the sigmoid of the sum of the product of A_k and the 
    * corresponding weight pointing to h_j
    * c) the derivative of the sigmoid of the sum of the product of h_j and the
    * corresponding weight pointing to F_1
    * d) the weight pointing from h_j to F_1
    * e) the value of A_k
    * f) the difference between the expected value and the output
    * 
    * This product is summed over all the models, with the final sum equal to the 
    * gradient.
    * 
    * @return a 2D array of doubles representing the gradients for each weight 
    */
   public double[][] calcChangeInWeights_kj()
   {
      double[][] changes = new double[NUM_INPUTS][NUM_HIDDEN];
      for (int jj = 0; jj < NUM_HIDDEN; jj++)
      {
         for (int kk = 0; kk < NUM_INPUTS; kk++)
         {
            for (int m = 0; m < NUM_MODELS; m++)
            {
               double change = LAMBDA;
               double sum_input = 0;
               for (int k = 0; k < NUM_INPUTS; k++)
               {
                  sum_input += inputNodes[k][m] * weights_kj[k][jj];
               }
               change *= sigmoidDeriv(sum_input);
               double sum_hidden = 0;
               for (int i = 0; i < NUM_OUTPUTS; i++)
               {
                  for (int j = 0; j < NUM_HIDDEN; j++)
                  {
                     sum_hidden += hiddenLayer[j] * weights_ji[j][i];
                  }
               }
               change *= sigmoidDeriv(sum_hidden);
               change *= weights_ji[jj][0];
               change *= (trainingSet[0][m] - output[0][m]);
               change *= inputNodes[kk][m];
               changes[kk][jj] += change;
            } // for (int m = 0; m < NUM_MODELS; m++)
         } //for (int kk = 0; kk < NUM_INPUTS; kk++)
      } // for (int jj = 0; jj < NUM_HIDDEN; jj++)
      return changes;
   }
   
   /*
    * Calculates the gradient for each of the weights between the j and i layers and
    * returns them in a 2D array.
    * 
    * The gradient is calculated through the product of 4 numbers:
    * 
    * a) LAMBDA, the reduction constant
    * b) the derivative of the sigmoid of the sum of the product of h_j and the
    * corresponding weight pointing to F_1
    * c) the value of h_j
    * d) the difference between the expected value and the output
    * 
    * This product is then summed over all the models, with the final sum equal to the
    * gradient.
    * 
    * @return a 2D array of doubles representing the gradients for each weight
    */
   public double[][] calcChangeInWeights_ji()
   {
      double[][] changes = new double[NUM_HIDDEN][NUM_OUTPUTS];
      for (int ii = 0; ii < NUM_OUTPUTS; ii++)
      {
         for (int jj = 0; jj < NUM_HIDDEN; jj++)
         {
            for (int m = 0; m < NUM_MODELS; m++)
            {
               double change = LAMBDA;
               change *= hiddenLayer[jj];
               change *= (trainingSet[0][m] - output[0][m]);
               double sum_hidden = 0;
               for (int i = 0; i < NUM_OUTPUTS; i++)
               {
                  for (int j = 0; j < NUM_HIDDEN; j++)
                  {
                     sum_hidden += hiddenLayer[j] * weights_ji[j][i];
                  }
               }
               change *= sigmoidDeriv(sum_hidden);
               changes[jj][ii] += change;
            } // for (int m = 0; m < NUM_MODELS; m++)
         } // for (int jj = 0; jj < NUM_HIDDEN; jj++)
      } // for (int ii = 0; ii < NUM_OUTPUTS; ii++)
      return changes;
   }
   
   /*
    * Changes each weight based off of its calculated gradient; increases each weight by
    * its corresponding weight gradient.  
    */
   public void applyWeightChanges()
   {
      double[][] changeInWeights_kj = calcChangeInWeights_kj();
      double[][] changeInWeights_ji = calcChangeInWeights_ji();
      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            weights_kj[k][j] = weights_kj[k][j] + changeInWeights_kj[k][j];
         }
      }
      for (int i = 0; i < NUM_OUTPUTS; i++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            weights_ji[j][i] = weights_ji[j][i] + changeInWeights_ji[j][i];
         }
      }
   }

   /*
    * Adjusts the weights and recalculates the output.  The method will continually 
    * adjust the weights until either the error falls under the error threshold or
    * the number of iterations exceeds the recalculation threshold.
    */
   public void adjustWeights()
   {
//      int i = 0;
//      while (errorFunction() > ERROR_THRESHOLD && i < RECALC_THRESHOLD)
//      {
//         applyWeightChanges();
//         calculateOutput();
//         i++;
//      }
//      System.out.println("\nAfter " + i + " iterations: \n");
   }
   /*
    * Given an input, returns the output of that value passed through
    * the sigmoid function.  Used in this case as the activation function
    * to keep the output between 0 and 1.
    * 
    * @param input the value to be passed to the sigmoid function
    * @return sigmoid(input)
    */
   public double sigmoid(double input)
   {
      double denom = 1 + Math.pow(Math.E, -1*input);
      return 1/denom;
   }
   
   /*
    * Given an input, returns the output of that value passed through
    * the derivative of the sigmoid function.  Used to determine the weight
    * changes by steepest descent.
    * 
    * @param input the value to be passed through the derivative of the sigmoid
    * @return sigmoid'(input)
    * 
    */
   public double sigmoidDeriv(double input)
   {
      double sigmoid = sigmoid(input);
      return sigmoid*(1-sigmoid);
   }
}
