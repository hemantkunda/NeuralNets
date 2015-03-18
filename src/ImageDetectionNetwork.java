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
 * @date December 3, 2014
 */
public class ImageDetectionNetwork 
{
   // Constant values of the network 
   private int numImages;
   private int numInputs;  // number of input nodes
   private int numHidden;                      // number of hidden nodes
   private int numOutputs;                                              // number of output nodes
   private int numTest;
   private int numTraining;
   String weightFilepath;                                     // name of the file that the weights are to be saved to
   String trainingSetFilepath;
   String configpath = "config.txt";
   String inputFilepath;
   String testFilepath;
   double lambda;                                                  // constant that controls the rate of change in the weights
   //Limit on # of back propagations
   int recalcThreshold;
   //Largest acceptable error value
   double errorThreshold;                                   // 5 e-6
   
   // Limits on initial weights
   int maxInitialWeight;
   int minInitialWeight;
   boolean training;
   // Storage of the layers and weights   
   DibDump[] trainingDumpers;
   DibDump[] testDumpers;
   String[] inputBMPS;
   String[] testBMPS;
   
   private int[][] inputNodes;
   private double[] hiddenLayer;     
   private double[][] weights_ji;
   private double[][] weights_kj;
   private double[][] output;
   private int[][] trainingSet;

   /*
    * Creates an instance of the network.  There are two possible modes for the 
    * network to be in: training mode or test mode.  This is defined by the boolean
    * flag parameter isTrainingCase. The storage arrays are initialized with either
    * the number of training cases or the number of test cases, depending on
    * isTrainingCase.  Then, the input information is loaded into either the 
    * training storage variable or the test storage variable, again depending on 
    * isTrainingCase.
    * 
    * @param isTrainingCase true if this network is being used to train weights for
    * a particular set of patterns; false otherwise
    */
   public ImageDetectionNetwork(boolean isTrainingCase)
   {
      training = isTrainingCase;
      importConfigData();
      initializeStorage();
      importTrainingSet();
      System.out.println(numImages);
      if (isTrainingCase)
      {
         for (int i = 0; i < numImages; i++)
         {
            DibDump d = new DibDump();
            d.inFileName = inputFilepath + inputBMPS[i] + ".bmp";
            d.importBitmapInfo();
            trainingDumpers[i] = d;
            inputNodes[i] = d.compressedImage;
         }
         flipInput();
      }
      else 
      {
         for (int i = 0; i < numTest; i++)
         {
            DibDump d = new DibDump();
            d.inFileName = testFilepath + testBMPS[i] + ".bmp";
            d.importBitmapInfo();
            testDumpers[i] = d;
            inputNodes[i] = d.compressedImage;
         }
         flipInput();
      }
   }
   
   
   /*
    * Creates a training network that generates random weights that are used to recognize
    * a variety of images defined in the config file.  These weights are then modified 
    * such that the network can use them to recognize any image in the training set 
    * and guess what an image outside of the training set is. These weights are then 
    * imported to a testing network, where they are used to recognize a test regiment 
    * defined in the config file.
    */
   
   public static void main(String[] args)
   {
      ImageDetectionNetwork network = new ImageDetectionNetwork(true);
      network.runNetwork();
      ImageDetectionNetwork testNet = new ImageDetectionNetwork(false);
      testNet.importWeights(new File(network.weightFilepath));
      testNet.verifyWeights();
      return;
   }
   
   /*
    * Initializes the storage variables based off of the specifications in the
    * config file.  This method is pointless unless the method importConfigData has
    * already been called.  
    */
   private void initializeStorage()
   {
      trainingDumpers = new DibDump[numTraining];
      testDumpers = new DibDump[numTest];
      inputNodes = new int[numImages][numInputs];
      output = new double[numOutputs][numImages];
      hiddenLayer = new double[numHidden];     
      weights_ji = new double[numHidden][numOutputs];
      weights_kj = new double[numInputs][numHidden];
      trainingSet = new int[numOutputs][numImages];
      return;
   }
   
   /*
    * This method performs the required functions of the test network: that is, test
    * the weights on a specified input.  That input is given outside of the method.
    * The network is evaluated using the imported weights, and the final output is 
    * printed to the console.
    */
   public void verifyWeights()
   {
      runOnTestCases();
      printFinalTestInfo();
      return;
   }
   
   /*
    * This method evaluates the network on each of the provided test images to see
    * if/how the network recognizes them.
    */
   public void runOnTestCases()
   {
      System.out.println();
      for (int m = 0; m < numImages; m++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            double sum = 0;
            for (int k = 0; k < numInputs; k++)
            {
               sum += inputNodes[k][m]*weights_kj[k][j];
            }
            hiddenLayer[j] = sigmoid(sum);
         }
         for (int i = 0; i < numOutputs; i++)
         {
            double sum = 0;
            for (int j = 0; j < numHidden; j++)
            {
               sum += hiddenLayer[j]*weights_ji[j][i];
            }
            output[i][m] = sigmoid(sum);
         }
      } // for (int m = 0; m < numImages; m++)
      return;
   }
   
   /*
    * A utility method that flips an array so that its rows become columns and its
    * columns become rows.  This is to be used only in the constructor, because the
    * initialization of the inputs is inverse to the initialization of the 
    * output layer.  To fix this problem, the input is flipped.  Any haphazard flipping
    * will likely create ArrayIndexOutOfBounds errors, so only use it in the constructor.
    */
   private void flipInput()
   {
      int[][] array = new int[numInputs][numImages];
      for (int r = 0; r < numImages; r++)
      {
         for (int c = 0; c < numInputs; c++)
         {
            array[c][r] = inputNodes[r][c];
         }
      }
      inputNodes = array;
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
    *    4) the network back-propogates to converge to an error specified in the config
    *    5) once the network has converged, the final information is printed to the 
    *    console
    *    6) the weights are saved to a text file
    */
   public void runNetwork()
   {
      //initializeInputs(0, numInputs - 1, 0, numImages - 1);
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
    * This method imports the constants of the network from a configuration file.
    * This method is designed to be compatible only with a configuration file that
    * follows this specification:
    * 
    * For most given lines, there are two elements: the name, and the value.
    * The two exceptions are be the TESTBMPS and INPUTBMPS lines.
    * 
    * The file's order should be:
    * 
    * 1) NUMTRAINING (number of training images)
    * 2) NUMINPUTS (number of nodes in the input layer - from the size of the images)
    * 3) NUMHIDDEN (number of nodes in the hidden layer)
    * 4) NUMOUTPUTS (number of nodes in the output layer)
    * 5) NUMTEST (number of testing images)
    * 6) TESTFILEPATH (file path to the folder containing the test images)
    * 7) TRAININGFILEPATH (file path to the folder containing the training images)
    * 8) WEIGHTSFILENAME (file path to the file where weights should be saved to or 
    *                       accessed)
    * 9) TRAININGSETFILENAME (file path to the file where the training set is located)
    * 10) LAMBDA (the constant used to amplify or diminish the changes made to the 
    *             weights during the training process)
    * 11) RECALCTHRESHOLD (the number of times the network will backpropagate before
    *                      stopping)
    * 12) ERRORTHRESHOLD (the maximum acceptable error for the network)
    * 13) MAXSTARTWEIGHT (the maximum initial value a weight can take on)
    * 14) MINSTARTWEIGHT (the minimum initial value a weight can take on)
    * 15) TESTBMPS (the names of each file in the collection of test images; all these
    *               files should be in the folder marked by TESTFILEPATH)
    * 16) INPUTBMPS (the names of each file in the collection of training images; all 
    *               these files should be in the folder marked by TRAININGFILEPATH)
    *                
    * Finally, the number of images that the network will deal with is initialized from
    * either numTest or numTraining, depending on what mode the network is in (defined
    * by the boolean flag training).
    */
   private void importConfigData()
   {
      File config = new File(configpath);
      try
      {
         Scanner in = new Scanner(config);
         in.next();
         numTraining = in.nextInt();
         in.next();
         numInputs = in.nextInt();
         in.next();
         numHidden = in.nextInt();
         in.next();
         numOutputs = in.nextInt();
         in.next();
         numTest = in.nextInt();
         in.next();
         testFilepath = in.next();
         in.next();
         inputFilepath = in.next();
         in.next();
         weightFilepath = in.next();
         in.next();
         trainingSetFilepath = in.next();
         in.next();
         lambda = in.nextDouble();
         in.next();
         recalcThreshold = in.nextInt();
         in.next();
         errorThreshold = in.nextDouble();
         in.next();
         maxInitialWeight = in.nextInt();
         in.next();
         minInitialWeight = in.nextInt();
         in.next();
         testBMPS = new String[numTest];
         for (int i = 0; i < numTest; i++)
         {
            testBMPS[i] = in.next();
         }
         in.next();
         inputBMPS = new String[numTraining];
         for (int i = 0; i < numTraining; i++)
         {
            inputBMPS[i] = in.next();
         }
         if (training)
         {
            numImages = numTraining;
         }
         else
         {
            numImages = numTest;
         }
      }
      catch (IOException e)
      {
         
      }
      return;  
   }
   
   /*
    * Imports the training set from the training file.  The training file is specified
    * in the config file.  The training file should have NUMOUTPUTS columns and 
    * NUMTRAINING rows.  The file is read and imported column major order. If the 
    * dimensions of the required training set do not match the dimensions of the 
    * information in the file, an error will be printed to the error console.
    */
   private void importTrainingSet()
   {
      File training = new File(trainingSetFilepath);
      try 
      {
         Scanner in = new Scanner(training);
         for (int m = 0; m < numTraining; m++)
         {
            for (int i = 0; i < numOutputs; i++)
            {
               if (in.hasNextInt())
               {
                  trainingSet[i][m] = in.nextInt();  
               }
               else
               {
                  System.err.print("Training Set has the wrong dimensions.");
                  return;
               }
            }
         }
         if (in.hasNextInt())
         {
            System.err.print("Training Set has the wrong dimensions.");
         }
         return;
      }
      
      catch (IOException e)
      {
         
      }
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
      double[][] kj = new double[numInputs][numHidden];
      double[][] ji = new double[numHidden][numOutputs];
      try
      {
         Scanner in = new Scanner(file);
         for (int k = 0; k < numInputs; k++)
         {
            for (int j = 0; j < numHidden; j++)
            {
               String str = in.next();
               double num = Double.parseDouble(str);
               kj[k][j] = num;
            }
         }
         
         for (int j = 0; j < numHidden; j++)
         {
            for (int i = 0; i < numOutputs; i++)
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
      File weights = new File(weightFilepath);
      try
      {  
         FileWriter out = new FileWriter(weights);
         for (int k = 0; k < numInputs; k++)
         {
            for (int j = 0; j < numHidden; j++)
            {
               double num = weights_kj[k][j];
               String str = (new Double(num)).toString();
               out.write(str + "\n");
            }
         }
         for (int j = 0; j < numHidden; j++)
         {
            for (int i = 0; i < numOutputs; i++)
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
    * Initializes each weight with a random value between maxInitialWeight and
    * minInitialWeight.
    */
   public void randomizeWeights()
   {
      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            weights_kj[k][j] = (Math.random() * maxInitialWeight + 
                                    Math.random() * minInitialWeight);
         }
      }
      
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            weights_ji[j][i] = (Math.random() * maxInitialWeight + 
                                    Math.random() * minInitialWeight);
         }
      }
      return;
   }
  
   /*
    * Loops through each image and prints out the corresponding initial outputs and 
    * the composite initial error.
    */
   public void printInitialInfo()
   {
      //System.out.println("Initial Weights: \n");
      //System.out.println("Weights_kj: \n");
      for (int j = 0; j < numHidden; j++)
      {
         for (int k = 0; k < numInputs; k++)
         {
            //System.out.println("\tw" + (k+1) + "-" + (j+1) + ": " + weights_kj[k][j]);
         }
      }
      
      //System.out.println("\nWeights_ji: \n");
      for (int i = 0; i < numOutputs; i++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            //System.out.println("\tw" + (j+1) + "-" + (i+1) + ": " + weights_ji[j][i]);
         }
      }
      
      System.out.println("\nInitial Outputs: ");
      for (int m = 0; m < numImages; m++)
      {
         System.out.println("Model " + (m+1) + ": ");
         System.out.println("Expected: ");
         for (int i = 0; i < numOutputs; i++)
         {
            System.out.println("\tF" + (i+1) + ": " + trainingSet[i][m]);
         }
         System.out.println();
         System.out.println("Actual: ");
         for (int i = 0; i < numOutputs; i++)
         {
            System.out.println("\tF" + (i+1) + ": " + output[i][m]);
         }
         System.out.println();
      } // for (int m = 0; m < numImages; m++)
      System.out.println("\nInitial Error: " + errorFunction());
      return;
   }
   
   /*
    * Loops through each image and prints out the final outputs and 
    * the composite final (converged) error.
    */
   public void printFinalInfo()
   {
      //System.out.println("Final Weights: \n");
      //System.out.println("Weights_kj: \n");
      for (int j = 0; j < numHidden; j++)
      {
         for (int k = 0; k < numInputs; k++)
         {
            //System.out.println("\tw" + (k+1) + "-" + (j+1) + ": " + weights_kj[k][j]);
         }
      }
      
      //System.out.println("\nWeights_ji: \n");
      for (int i = 0; i < numOutputs; i++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            //System.out.println("\tw" + (j+1) + "-" + (i+1) + ": " + weights_ji[j][i]);
         }
      }
      System.out.println("\nFinal Outputs: ");
      for (int m = 0; m < numImages; m++)
      {
         System.out.println("Model " + (m+1) + ": ");
         System.out.println("Expected: ");
         for (int i = 0; i < numOutputs; i++)
         {
            System.out.println("\tF" + (i+1) + ": " + trainingSet[i][m]);
         }
         System.out.println();
         System.out.println("Actual: ");
         for (int i = 0; i < numOutputs; i++)
         {
            System.out.println("\tF" + (i+1) + ": " + output[i][m]);
         }
         System.out.println();
      }
      System.out.println("\nFinal Error: " + errorFunction());
      return;
   }
   
   /*
    * Prints out the outputs of the network after evaluating the images and generating
    * the output.  This method does not actually generate the output; it must be 
    * done beforehand, so calling this method before evaluating the test network is 
    * rather pointless.
    */
   public void printFinalTestInfo()
   {
      for (int i = 0; i < numImages; i++)
      {
         System.out.println(testBMPS[i]+ ": ");
         for (int j = 0; j < numOutputs; j++)
         {
            System.out.println(output[j][i] + "   ");
         }
         System.out.println("\n");
      }
      return;
   }
   
   /*
    * Continually runs the back propagation until the error falls below the
    * error threshold or the back propagation has run more times than the
    * recalculation threshold.
    * 
    * While the number of recalculations is less than the recalc threshold or 
    * the error is larger than the acceptable error, the back propagation is calculated,
    * and the outputs for each image and the composite error are printed to reflect
    * the network's progress.
    */
   public void backPropagate()
   {
      int u = 0;
      int v = 0;
      while (errorFunction() > errorThreshold && u < recalcThreshold)
      {
         calculateOutput(v);
         singleBackPropagation(v);
         u++;
         v++;
         if (v >= numImages)
         {
            v = 0;
         }
         System.out.println("\n\nRound " + u);
         for (int m = 0; m < numImages; m++)
         {
            System.out.println("Model " + (m+1) + ": ");
            System.out.println("Expected: ");
            for (int i = 0; i < numOutputs; i++)
            {
               System.out.println("\tF" + (i+1) + ": " + trainingSet[i][m]);
            }
            System.out.println();
            System.out.println("Actual: ");
            for (int i = 0; i < numOutputs; i++)
            {
               System.out.println("\tF" + (i+1) + ": " + output[i][m]);
            }
            System.out.println();
         } // for (int m = 0; m < numImages; m++)
         System.out.println("Trial " + u + " Error: " + errorFunction());
      }
      System.out.println("\nAfter " + u + " iterations: \n");
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
      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            double Omega_j = 0;
            for (int i = 0; i < numOutputs; i++)
            {
               double psi_i = psi_i(i, m);
               double deltaWeights_ji = lambda * hiddenLayer[j] * psi_i;
               weights_ji[j][i] = weights_ji[j][i] + deltaWeights_ji;
               Omega_j += psi_i * weights_ji[j][i];      

            }
            double Theta_j = Theta_j(j, m);
            double Psi_j = Psi_j(Omega_j, Theta_j);
            double deltaWeights_kj = lambda * inputNodes[k][m] * Psi_j;
            weights_kj[k][j] = weights_kj[k][j] + deltaWeights_kj;
            //System.out.println("j: " + j);
         } // for (int j = 0; j < numHidden; j++)
         //System.out.println("k: " + k);
      } // for (int k = 0; k < numInputs; k++)
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
      for (int m = 0; m < numImages; m++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            double sum = 0;
            for (int k = 0; k < numInputs; k++)
            {
               sum += inputNodes[k][m]*weights_kj[k][j];
            }
            hiddenLayer[j] = sigmoid(sum);
         }
         for (int i = 0; i < numOutputs; i++)
         {
            double sum = 0;
            for (int j = 0; j < numHidden; j++)
            {
               sum += hiddenLayer[j]*weights_ji[j][i];
            }
            output[i][m] = sigmoid(sum);
         }
      } // for (int m = 0; m < numImages; m++)
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
      for (int j = 0; j < numHidden; j++)
      {
         double sum = 0;
         for (int k = 0; k < numInputs; k++)
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
      for (int i = 0; i < numOutputs; i++)
      {
         double sum = 0;
         for (int j = 0; j < numHidden; j++)
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
    * over all the numImages.
    * 
    * @return the value of the error function
    */
   public double errorFunction()
   {
      double sum = 0;
      for (int m = 0; m < numImages; m++)
      {
         for (int i = 0; i < numOutputs; i++)
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
    * @param omega_i the value of omega_i, the sum over the numImages
    * of the difference between the training value and output value
    * @param Theta_i the value of Theta_i, the sum over the numImages of the 
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

      for (int j = 0; j < numHidden; j++)
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
      for (int k = 0; k < numInputs; k++)
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
    * a) lambda, the reduction constant
    * b) the derivative of the sigmoid of the sum of the product of A_k and the 
    * corresponding weight pointing to h_j
    * c) the derivative of the sigmoid of the sum of the product of h_j and the
    * corresponding weight pointing to F_1
    * d) the weight pointing from h_j to F_1
    * e) the value of A_k
    * f) the difference between the expected value and the output
    * 
    * This product is summed over all the numImages, with the final sum equal to the 
    * gradient.
    * 
    * @return a 2D array of doubles representing the gradients for each weight 
    */
   public double[][] calcChangeInWeights_kj()
   {
      double[][] changes = new double[numInputs][numHidden];
      for (int jj = 0; jj < numHidden; jj++)
      {
         for (int kk = 0; kk < numInputs; kk++)
         {
            for (int m = 0; m < numImages; m++)
            {
               double change = lambda;
               double sum_input = 0;
               for (int k = 0; k < numInputs; k++)
               {
                  sum_input += inputNodes[k][m] * weights_kj[k][jj];
               }
               change *= sigmoidDeriv(sum_input);
               double sum_hidden = 0;
               for (int i = 0; i < numOutputs; i++)
               {
                  for (int j = 0; j < numHidden; j++)
                  {
                     sum_hidden += hiddenLayer[j] * weights_ji[j][i];
                  }
               }
               change *= sigmoidDeriv(sum_hidden);
               change *= weights_ji[jj][0];
               change *= (trainingSet[0][m] - output[0][m]);
               change *= inputNodes[kk][m];
               changes[kk][jj] += change;
            } // for (int m = 0; m < numImages; m++)
         } //for (int kk = 0; kk < numInputs; kk++)
      } // for (int jj = 0; jj < numHidden; jj++)
      return changes;
   }
   
   /*
    * Calculates the gradient for each of the weights between the j and i layers and
    * returns them in a 2D array.
    * 
    * The gradient is calculated through the product of 4 numbers:
    * 
    * a) lambda, the reduction constant
    * b) the derivative of the sigmoid of the sum of the product of h_j and the
    * corresponding weight pointing to F_1
    * c) the value of h_j
    * d) the difference between the expected value and the output
    * 
    * This product is then summed over all the numImages, with the final sum equal to the
    * gradient.
    * 
    * @return a 2D array of doubles representing the gradients for each weight
    */
   public double[][] calcChangeInWeights_ji()
   {
      double[][] changes = new double[numHidden][numOutputs];
      for (int ii = 0; ii < numOutputs; ii++)
      {
         for (int jj = 0; jj < numHidden; jj++)
         {
            for (int m = 0; m < numImages; m++)
            {
               double change = lambda;
               change *= hiddenLayer[jj];
               change *= (trainingSet[0][m] - output[0][m]);
               double sum_hidden = 0;
               for (int i = 0; i < numOutputs; i++)
               {
                  for (int j = 0; j < numHidden; j++)
                  {
                     sum_hidden += hiddenLayer[j] * weights_ji[j][i];
                  }
               }
               change *= sigmoidDeriv(sum_hidden);
               changes[jj][ii] += change;
            } // for (int m = 0; m < numImages; m++)
         } // for (int jj = 0; jj < numHidden; jj++)
      } // for (int ii = 0; ii < numOutputs; ii++)
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
      for (int j = 0; j < numHidden; j++)
      {
         for (int k = 0; k < numInputs; k++)
         {
            weights_kj[k][j] = weights_kj[k][j] + changeInWeights_kj[k][j];
         }
      }
      for (int i = 0; i < numOutputs; i++)
      {
         for (int j = 0; j < numHidden; j++)
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
      while (errorFunction() > errorThreshold && i < recalcThreshold)
      {
         applyWeightChanges();
         calculateOutput(m);
         i++;
         m++;
         if (m >= numImages)
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
