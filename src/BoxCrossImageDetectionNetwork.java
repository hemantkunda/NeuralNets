import java.io.File;
import java.io.IOException;
import java.io.FileWriter;
import java.util.Scanner;
/**
 * This class models a perceptron, distinguishing between 
 * different shapes represented by arrays of 1's and 0's.  The perceptron uses a 
 * back-propagation strategy to train the weights in order to recognize the different
 * shapes. The weights are considered trained once the error falls below 1 e-6.    
 * 
 * The number of inputs is determined by the pattern itself; patterns that fall
 * in a 4x4 matrix, for example, would correspond to 16 input nodes.  The number of hidden
 * nodes is variable, while the number of outputs is two (or the number of patterns that
 * the perceptron is trained to recognize).
 * Each output should correspond to whether or not a certain pattern is recognized. 
 * 
 * @author Hemant Kunda
 * @date October 5, 2014
 */
public class BoxCrossImageDetectionNetwork 
{
   // Constant values of the network 
   final int NUM_INPUTS = 16;                         // number of input nodes
   final int NUM_HIDDEN = 70;                         // number of hidden nodes
   final int NUM_OUTPUTS = 2;                         // number of output nodes
   final String FILE_PATH = "weights.txt"; // name of the file that the weights are to be saved to
   final int NUM_WEIGHTS_KJ = NUM_INPUTS*NUM_HIDDEN;  // number of weights in the kj layer
   final int NUM_WEIGHTS_JI = NUM_HIDDEN*NUM_OUTPUTS; // number of weights in the ji layer
   final int NUM_MODELS;                              // number of models - initialized in the constructor
   static final double LAMBDA = 1;                    // constant that controls the rate of change in the weights
   //Limit on # of back propagations
   static final int RECALC_THRESHOLD = 1000000;
   //Largest acceptable error value
   static final double ERROR_THRESHOLD = .000001;  
   
   // Limits on initial weights
   static final double MAX_INITIAL_WEIGHT = 5;
   static final double MIN_INITIAL_WEIGHT = -5;
   
   // Storage of the layers and weights

   private int[][] inputNodes;
   private int[][] testInputNodes = {{1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1},
                                     {1}};
   /*
    * column 1: box only
    * column 2: cross only
    */
   private int[][] trainingInputNodes = {{1,1},               
                                         {1,0},
                                         {1,0},
                                         {1,1},
                                         {1,0},
                                         {0,1},
                                         {0,1},
                                         {1,0},
                                         {1,0},
                                         {0,1},
                                         {0,1},
                                         {1,0},
                                         {1,1},
                                         {1,0},
                                         {1,0},
                                         {1,1}};
   private double[] hiddenLayer;     
   private double[][] weights_ji = new double[NUM_HIDDEN][NUM_OUTPUTS];
   private double[][] weights_kj = new double[NUM_INPUTS][NUM_HIDDEN];
   private double[][] output;
   private int[][] trainingSet = {{1,0},
                                  {0,1}}; // expected output during training

   /*
    * Creates an instance of the network.  There are two possible modes for the 
    * network to be in: training mode or test mode.  This is defined by the first
    * parameter.  The number of training models and test models is also specified; the
    * number of models is set to one of those values depending on the mode that
    * the network is in.
    * 
    * @param isTrainingCase true if this network is being used to train weights for
    * a particular set of patterns; false otherwise
    * @param numTrainingModels the number of scenarios that the network is to train the
    * weights on
    * @param numTestModels the number of scenarios that the network is to test the weights
    * with
    */
   public BoxCrossImageDetectionNetwork(boolean isTrainingCase, int numTrainingModels, int numTestModels)
   {
      if (isTrainingCase)
      {
         inputNodes = trainingInputNodes;
         NUM_MODELS = numTrainingModels;
      }
      else 
      {
         inputNodes = testInputNodes;
         NUM_MODELS = numTestModels;
      }
      hiddenLayer = new double[NUM_HIDDEN];
      output = new double[NUM_OUTPUTS][NUM_MODELS];
   }
   
   /*
    * Creates a training network that generates random weights that are used to recognize
    * two different patterns (in this case, a box and a cross).  These weights are then
    * modified such that the network can use them to recognize either pattern 1 or pattern
    * 2 (either a box or a cross).  These weights are then imported to a testing network,
    * where they are used to recognize an input containing both pattern 1 and pattern 2
    * (both a box and a cross).
    */
   public static void main(String[] args)
   {
      BoxCrossImageDetectionNetwork network = new BoxCrossImageDetectionNetwork(true, 2, 1);
      network.runNetwork();
      BoxCrossImageDetectionNetwork testNet = new BoxCrossImageDetectionNetwork(false, 2, 1);
      testNet.importWeights(new File(testNet.FILE_PATH));
      testNet.calculateInitialOutput();
      testNet.printFinalTestInfo();
      return;
   }
   
   /*
    * Accessor method for the weights in the kj layer.
    * 
    * @return the weights_kj
    */
   public double[][] getWeights_kj()
   {
      return weights_kj;
   }
   
   /*
    * Accessor method for the weights in the ji layer.
    * 
    * @return the weights_ji
    */
   public double[][] getWeights_ji()
   {
      return weights_ji;
   }
   
   /*
    * Mutator method for the weights in the kj layer.
    * 
    * @param newWeights_kj the set of weights to replace the current weights in the kj 
    * layer.
    * 
    * Precondition: newWeights_kj have the same dimensions as weights_kj.
    */
   public void setWeights_kj(double[][] newWeights_kj)
   {
      weights_kj = newWeights_kj;
      return;
   }
   
   /*
    * Mutator method for the weights in the ji layer.
    * 
    * @param newWeights_ji the set of weights to replace the current weights in the ji 
    * layer.
    * 
    * Precondition: newWeights_ji have the same dimensions as weights_ji.
    */
   public void setWeights_ji(double[][] newWeights_ji)
   {
      weights_ji = newWeights_ji;
      return;
   }
   
   /*
    * Runs the network by performing key steps.  In the case of this particular network:
    *   
    *    1) weights are randomly generated
    *    2) the initial output is calculated
    *    3) the initial information is printed to the console
    *    4) the network back-propogates to converge to an error of 0
    *    5) once the network has converged, the final information is printed to the 
    *    console
    *    6) the weights are saved to a text file
    */
   public void runNetwork()
   {
      //initializeInputs(0, NUM_INPUTS - 1, 0, NUM_MODELS - 1);
      //createXORTrainingSet();
      //createXNORTrainingSet();
      randomizeWeights();
      calculateInitialOutput();
      printInitialInfo();
      //adjustWeights();
      backPropagate();
      printFinalInfo();
      saveWeights();
      return;
   }
   
   /*
    * Imports weights from a file.
    * 
    * This file must be formatted as follows:
    *    1) weights_kj followed by weights_ji, row by row
    *    2) one weight per line
    * 
    * @param file the file containing the weights. 
    */
   public void importWeights(File file)
   {
      double[][] kj = new double[NUM_INPUTS][NUM_HIDDEN];
      double[][] ji = new double[NUM_HIDDEN][NUM_OUTPUTS];
      try
      {
         Scanner in = new Scanner(file);
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            for (int j = 0; j < NUM_HIDDEN; j++)
            {
               String str = in.next();
               double num = Double.parseDouble(str);
               kj[k][j] = num;
            }
         }
         
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            for (int i = 0; i < NUM_OUTPUTS; i++)
            {
               String str = in.next();
               double num = Double.parseDouble(str);
               ji[j][i] = num;
            }
         }
         setWeights_kj(kj);
         setWeights_ji(ji);
      } // try
      catch (IOException e)
      {
         
      }
      return;
   }
   
   /*
    * Saves the weights to a file.
    * 
    * The weights are saved as follows:
    *    1) weights_kj followed by weights_ji
    *    2) row by row
    *    3) each line contains 1 weight
    */
   public void saveWeights()
   {
      File weights = new File(FILE_PATH);
      try
      {  
         FileWriter out = new FileWriter(weights);
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            for (int j = 0; j < NUM_HIDDEN; j++)
            {
               double num = weights_kj[k][j];
               String str = (new Double(num)).toString();
               out.write(str + "\n");
            }
         }
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            for (int i = 0; i < NUM_OUTPUTS; i++)
            {
               double num = weights_ji[j][i];
               String str = (new Double(num)).toString();
               out.write(str + "\n");
            }
         }
         out.close();
      } // try
      catch (IOException e)
      {
         
      }
      return;
   }
   
   /*
    * Initializes each weight with a random value between MAX_INITIAL_WEIGHT and
    * MIN_INITIAL_WEIGHT.
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
      return;
   }

   /*
    * Creates the XOR training set in the first layer of the training set variable.
    * The output of XOR with multiple outputs is determined by the number of 1's for 
    * two reasons:
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
      return;
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
      return;
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
      return;
   }
   
   /*
    * Prints out the initial weights, initial outputs and initial error.
    */
   public void printInitialInfo()
   {
      //System.out.println("Initial Weights: \n");
      //System.out.println("Weights_kj: \n");
      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            //System.out.println("\tw" + (k+1) + "-" + (j+1) + ": " + weights_kj[k][j]);
         }
      }
      
      //System.out.println("\nWeights_ji: \n");
      for (int i = 0; i < NUM_OUTPUTS; i++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            //System.out.println("\tw" + (j+1) + "-" + (i+1) + ": " + weights_ji[j][i]);
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
      } // for (int m = 0; m < NUM_MODELS; m++)
      System.out.println("\nInitial Error: " + errorFunction());
      return;
   }
   
   /*
    * Prints out the final weights, final outputs and final (converged) error.
    */
   public void printFinalInfo()
   {
      //System.out.println("Final Weights: \n");
      //System.out.println("Weights_kj: \n");
      for (int j = 0; j < NUM_HIDDEN; j++)
      {
         for (int k = 0; k < NUM_INPUTS; k++)
         {
            //System.out.println("\tw" + (k+1) + "-" + (j+1) + ": " + weights_kj[k][j]);
         }
      }
      
      //System.out.println("\nWeights_ji: \n");
      for (int i = 0; i < NUM_OUTPUTS; i++)
      {
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            //System.out.println("\tw" + (j+1) + "-" + (i+1) + ": " + weights_ji[j][i]);
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
      return;
   }
   
   /*
    * Prints out the output for the box and cross.  The output should be 1 if it uses 
    * weights that have converged via back-propagation in a training network.  Practical 
    * use for this method only exists if the network is a testing network. 
    */
   public void printFinalTestInfo()
   {
      System.out.println("\n\nBox: " + output[0][0]);
      System.out.println("Cross: " + output[1][0]);
      return;
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
      return;
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
         } // for (int j = 0; j < NUM_HIDDEN; j++)
      } // for (int k = 0; k < NUM_INPUTS; k++)
      return;
   }
   
   /*
    * Calculates the values of the hidden layer, then calculates the values of the
    * output.
    */
   public void calculateOutput(int m)
   {
      sumIntoHidden(m);
      sumIntoOutput(m);
      return;
   }
   
   /*
    * Calculates the output for each model.  This is achieved by calculating the values
    * of the hidden layer nodes (see method sumIntoHidden for more info), then calculating
    * the values of the output nodes (see method sumIntoOutput).
    */
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
      } // for (int m = 0; m < NUM_MODELS; m++)
      return;
   }
   
   /*
    * Loops over each model, calculating the value of each hidden layer node
    * by summing the product of each weight connected to the node and the input
    * value attached to that weight and passing that value through the sigmoid
    * function. 
    */
   public void sumIntoHidden(int m)
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
      return;
   }
   /*
    * Loops over each model, calculating the value of each output node by summing
    * the product of each weight connected to the output node and the value of the
    * hidden node attached to that weight and passing that value through the 
    * sigmoid function.
    */
   public void sumIntoOutput(int m)
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
      return;
   }
   
   /*
    * Calculates the error function.  The error function is half of
    * the square of the output values subtracted from the training values, summed 
    * over all the NUM_MODELS.
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
    * @param omega_i the value of omega_i, the sum over the NUM_MODELS
    * of the difference between the training value and output value
    * @param Theta_i the value of Theta_i, the sum over the NUM_MODELS of the 
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
    * This product is summed over all the NUM_MODELS, with the final sum equal to the 
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
    * This product is then summed over all the NUM_MODELS, with the final sum equal to the
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
      return;
   }

   /*
    * Adjusts the weights and recalculates the output.  The method will continually 
    * adjust the weights until either the error falls under the error threshold or
    * the number of iterations exceeds the recalculation threshold.
    */
   public void adjustWeights()
   {
      int i = 0;
      int m = 0;
      while (errorFunction() > ERROR_THRESHOLD && i < RECALC_THRESHOLD)
      {
         applyWeightChanges();
         calculateOutput(m);
         i++;
         m++;
         if (m >= NUM_MODELS)
         {
            m = 0;
         }
      }
      System.out.println("\nAfter " + i + " iterations: \n");
      return;
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
      double denom = 1 + Math.exp(-1*input);
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
      double sig = sigmoid(input);
      return sig*(1-sig);
   }
}
