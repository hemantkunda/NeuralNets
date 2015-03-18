/**
 * This class models a basic multilayer perceptron with a variable amount of 
 * hidden nodes, 2 input nodes, and 1 output node.
 * 
 * @author Hemant Kunda
 * @date September 11, 2014
 */
public class BasicNetwork {
   // Final values of the network 
   static final int NUM_INPUTS = 2;
   static final int NUM_HIDDEN = 4;
   static final int NUM_OUTPUTS = 1;
   static final int NUM_WEIGHTS_KJ = NUM_INPUTS*NUM_HIDDEN;
   static final int NUM_WEIGHTS_JI = NUM_HIDDEN*NUM_OUTPUTS;
   static final int NUM_MODELS = 4;
   static final double LAMBDA = 1;
   static final int RECALC_THRESHOLD = 400000;
   static final double ERROR_THRESHOLD = .001;
   
   // Limits on initial weights
   static final double MAX_INITIAL_WEIGHT = 5;
   static final double MIN_INITIAL_WEIGHT = -5;
   
   // Storage of the layers and weights
   private int[][] inputNodes = {{1,0,1,0},
                                 {1,1,0,0}};
   private double[][] hiddenLayer = new double[NUM_HIDDEN][NUM_MODELS];
   public double[][] weights_ji = new double[NUM_HIDDEN][NUM_OUTPUTS];
   public double[][] weights_kj = new double[NUM_INPUTS][NUM_HIDDEN];
   public double[][] output = new double[NUM_OUTPUTS][NUM_MODELS];
   public int[] trainingSet = new int[NUM_MODELS];
   
   public BasicNetwork()
   {
      
   }
   
   /*
    * Initializes the weights with random values, prints out the initial information for 
    * the perceptron, converges the error function, and prints out the final data.
    */
   public static void main(String[] args)
   {
      BasicNetwork bob = new BasicNetwork();
      for (int k = 0; k < NUM_INPUTS; k++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            bob.weights_kj[k][j] = (Math.random() * MAX_INITIAL_WEIGHT + 
                                    Math.random() * MIN_INITIAL_WEIGHT);
         }
      }
      
      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         for (int i = 0; i < NUM_OUTPUTS; i++)
         {
            bob.weights_ji[j][i] = (Math.random() * MAX_INITIAL_WEIGHT + 
                                    Math.random() * MIN_INITIAL_WEIGHT);
         }
      }
      bob.createTrainingSet();
      bob.runNetwork();
   }
   
   /*
    * Calculates the initial outputs, prints out the initial information, adjusts
    * the weights based off the gradients, and prints out the final (converged) 
    * information.
    */
   public void runNetwork()
   {
      calculateOutput();
      printInitialInfo();
      adjustWeights();
      printFinalInfo();
   }
   
   public void createTrainingSet()
   {
      for (int m = 0; m < NUM_MODELS; m++)
      {
         int trainingVal = 0;
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            trainingVal += inputNodes[k][m];
         }
         trainingVal = trainingVal % 2;
         trainingSet[m] = trainingVal;
      }
   }
   
   /*
    * Prints out the initial weights and initial error.
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
      System.out.println("\nInitial Error: " + errorFunction());
   }
   
   /*
    * Prints out the final weights and final (converged) error.
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
      System.out.println("\nFinal Error: " + errorFunction());
   }
   
   /*
    * Calculates the values of the hidden layer, then calculates the values of the
    * output.
    */
   public void calculateOutput()
   {
      sumIntoHidden();
      sumIntoOutput();
   }
   
   /*
    * Loops over each model, calculating the value of each hidden layer node
    * by summing the product of each weight connected to the node and the input
    * value attached to that weight and passing that value through the sigmoid
    * function. 
    */
   public void sumIntoHidden()
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
            hiddenLayer[j][m] = sigmoid(sum);
         }
      }
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
               sum += hiddenLayer[j][m]*weights_ji[j][i];
            }
            output[i][m] = sigmoid(sum);
         }
      }
   }
   
   /*
    * Returns the value of the error function.  The error function is half of
    * the square of the actual value subtracted from the training value, summed 
    * over all the models.
    * 
    * @return the value of the error function
    */
   public double errorFunction()
   {
      double sum = 0;
      for (int i = 0; i < NUM_MODELS; i++)
      {
         sum += (trainingSet[i] - output[0][i]) * (trainingSet[i] - output[0][i]);
      }
      return .5 * sum;
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
                     sum_hidden += hiddenLayer[j][m] * weights_ji[j][i];
                  }
               }
               change *= sigmoidDeriv(sum_hidden);
               change *= weights_ji[jj][0];
               change *= (trainingSet[m] - output[0][m]);
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
               change *= hiddenLayer[jj][m];
               change *= (trainingSet[m] - output[0][m]);
               double sum_hidden = 0;
               for (int i = 0; i < NUM_OUTPUTS; i++)
               {
                  for (int j = 0; j < NUM_HIDDEN; j++)
                  {
                     sum_hidden += hiddenLayer[j][m] * weights_ji[j][i];
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
      int i = 0;
      while (errorFunction() > ERROR_THRESHOLD && i < RECALC_THRESHOLD)
      {
         applyWeightChanges();
         calculateOutput();
         i++;
      }
      System.out.println("\nAfter " + i + " iterations: \n");
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
      return sigmoid(input)*(1-sigmoid(input));
   }
   
}
