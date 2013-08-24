
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvSize;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_highgui.*;
import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_contrib.*;
import static com.googlecode.javacv.cpp.opencv_objdetect.CV_HAAR_FIND_BIGGEST_OBJECT;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.CascadeClassifier;
public class FaceEx {
	static int pre_Process =0;
	List emotionNames = new ArrayList<>();
	java.io.File file = new java.io.File("");
	String  abspath=file.getAbsolutePath();
	final List classOrder = new ArrayList<>();
	int nFaces=0;
	int nPersons;
	int rows;
	int pred;
	double conf;
	static FaceRecognizer classifier;
	FaceRecognizer classifier1;
	FaceRecognizer classifier2;
	FaceRecognizer classifier3;
	FaceRecognizer classifier4;
	FaceRecognizer classifier5;
	FaceRecognizer classifier6;
	FaceRecognizer classifier7;
	FaceRecognizer classifier8;
	FaceRecognizer classifier9;
	FaceRecognizer classifier10;
	
	
	public FaceEx(){
		classifier = createFisherFaceRecognizer();
		classifier1 = createFisherFaceRecognizer();
		classifier2 = createFisherFaceRecognizer();
		classifier3 = createFisherFaceRecognizer();
		classifier4 = createFisherFaceRecognizer();
		classifier5 = createFisherFaceRecognizer();
		classifier6 = createFisherFaceRecognizer();
		classifier7 = createFisherFaceRecognizer();
		classifier8 = createFisherFaceRecognizer();
		classifier9 = createFisherFaceRecognizer();
		classifier10 = createFisherFaceRecognizer();
		
	}
	
	
	static CvMat norm_0_255(CvMat _src){


		CvMat src = _src;


		CvMat dst=cvCreateMat(src.rows(),src.cols(),src.type());



		switch(_src.channels()){

		case 1:
			cvNormalize(
					_src, // src (CvArr)
					dst, // dst (CvArr)
					0, // a
					255, // b
					NORM_MINMAX, // norm_type
					null); // mask
			break;
		case 3:
			cvNormalize(
					_src, // src (CvArr)
					dst, // dst (CvArr)
					0, // a
					255, // b
					NORM_MINMAX, // norm_type
					null); // mask
			break;
		default:
			dst=src.clone();
			break;
		}
		return dst;
	}
	
	private IplImage[] loadImage() {

		BufferedReader imgListFile;
		IplImage[] faceImgArr;
		String imgFilename;
		int iFace = 0;
		int i;
		try{
			imgListFile = new BufferedReader(new FileReader(abspath+"/Exrec.txt"));
			while (true) {
				final String line = imgListFile.readLine();
				if (line == null || line.isEmpty()) {
					break;
				}
				nFaces++;
			}

			imgListFile = new BufferedReader(new FileReader(abspath+"/Exrec.txt"));	
			faceImgArr = new IplImage[nFaces];
			classOrder.clear();
			emotionNames.clear();        // Make sure it starts as empty.
			nPersons = 0;

			for (iFace = 0; iFace < nFaces; iFace++) {
				String personName;
				String sPersonName;
				int personNumber;

				// read person number (beginning with 1), their name and the image filename.
				final String line = imgListFile.readLine();
				if (line.isEmpty()) {
					break;
				}
				final String[] tokens = line.split(" ");
				personNumber = Integer.parseInt(tokens[0]);
				personName = tokens[1];
				imgFilename =  tokens[2];
				sPersonName = personName;
				classOrder.add(personNumber);

				// Keep the data
				// Check if a new person is being loaded.
				
				preProcess(imgFilename,pre_Process);

				if (personNumber > nPersons) {
					// Allocate memory for the extra person (or possibly multiple), using this new person's name.
					emotionNames.add(sPersonName);

					nPersons = personNumber;


				}


				faceImgArr[iFace]=cvLoadImage(imgFilename);
				if (faceImgArr[iFace] == null) {
					throw new RuntimeException("Can't load image from " + imgFilename);
				}
			}
			imgListFile.close();
		}
		catch(IOException ex) {
			throw new RuntimeException(ex);
		}

		final StringBuilder stringBuilder = new StringBuilder();
		stringBuilder.append("People: ");
		if (nPersons > 0) {
			stringBuilder.append("<").append(emotionNames.get(0)).append(">");
		}
		for (i = 1; i < nPersons && i < emotionNames.size(); i++) {
			stringBuilder.append(", <").append(emotionNames.get(i)).append(">");
		}


		return faceImgArr;
	}
	public void preProcess(String s,int u){


		int avg_x1=0,avg_y1=0,avg_x2=0,avg_y2=0;
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		CascadeClassifier faceDetector = new CascadeClassifier("lbpcascade_frontalface.xml");
		CascadeClassifier eyeDetector = new CascadeClassifier("haarcascade_eye.xml");

		Mat image = Highgui.imread(s,Highgui.IMREAD_GRAYSCALE);

		//String[] tokens = phrase.split(delims);

		
		Boolean x=Highgui.imwrite(s,image);
		
		MatOfRect faceDetections = new MatOfRect();
		faceDetector.detectMultiScale(image, faceDetections,1.1, 5,CV_HAAR_FIND_BIGGEST_OBJECT, new Size(image.rows()/6,image.cols()/6),image.size());//image,objects,scalefactor,minNeighbors,flags,minsize,maxsize

		if(faceDetections.toArray().length == 1){
			for (Rect rect : faceDetections.toArray()) {

				Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));




				Mat face = Highgui.imread(s).submat(rect.y,rect.y+rect.height,rect.x,rect.x+rect.width);
				Highgui.imwrite(s, face);
			}

			image = Highgui.imread(s,Highgui.IMREAD_GRAYSCALE);
			MatOfRect eyeDetections = new MatOfRect();


			eyeDetector.detectMultiScale(image,eyeDetections,1.1,30,0, new Size(24,24),image.size());



			if (eyeDetections.toArray().length ==2){
				int i=0;
				for (Rect rect : eyeDetections.toArray()) {

					Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
					//Highgui.imwrite(s, image);
					if (i==0){
						int diff =0;
						diff=((rect.x+rect.width)-rect.x)/2;
						avg_x1 = rect.x + diff;
						diff = ((rect.y+rect.height)-rect.y)/2;
						avg_y1=rect.y+diff;

					}
					else{
						int diff =0;
						diff=((rect.x+rect.width)-rect.x)/2;
						avg_x2 = rect.x + diff;
						diff = ((rect.y+rect.height)-rect.y)/2;
						avg_y2=rect.y+diff;


					}
					i++;
				}	



				int tempx=0,tempy=0;
				if(avg_x1>avg_x2){
					tempx=avg_x1;
					tempy=avg_y1;
					avg_x1=avg_x2;
					avg_y1=avg_y2;
					avg_x2=tempx;
					avg_y2=tempy;

				}
			}
			else{
				try {
					BufferedImage bimg = ImageIO.read(new File(s));
					int width          = bimg.getWidth();
					int height         = bimg.getHeight();
					avg_y1=avg_y2=(int)(0.3*height);
					avg_x1 = (int)(0.25*width);
					avg_x2 = (int)(0.75 * width);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}  
			}

		}
		else{
			try {
				BufferedImage bimg = ImageIO.read(new File(s));
				int width          = bimg.getWidth();
				int height         = bimg.getHeight();
				avg_y1=avg_y2=(int)(0.3*height);
				avg_x1 = (int)(0.25*width);
				avg_x2 = (int)(0.75 * width);
				image = Highgui.imread(s,Highgui.IMREAD_GRAYSCALE);
				Highgui.imwrite(s, image);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}  	
		}

		String cmd= null;
		String[] command = new String[7];
		command[0]="python";
		command[1]=abspath + "/align.py";
		command[2] =avg_x1 + "";
		command[3] =avg_y1 + "";
		command[4] =avg_x2 + "";
		command[5] =avg_y2 + "";
		command[6]=s+"";
		Process process=null;

		try {
			
			process = Runtime.getRuntime().exec(command);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			process.waitFor();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		process.destroy();
        Mat classifier1_image,classifier2_image,classifier3_image;
        image=Highgui.imread(s);
        if(u==0){
        	classifier1_image = Highgui.imread(s).submat(1, 68, 0, image.cols());
        	Highgui.imwrite(s, classifier1_image);
        }
        else if (u==1){
        	classifier2_image = Highgui.imread(s).submat(69, 125, 0, image.cols());
        	Highgui.imwrite(s, classifier2_image);
        }
        else if (u==2){
        	classifier3_image = Highgui.imread(s).submat(126, 187, 0, image.cols());
        	Highgui.imwrite(s, classifier3_image);
        }
		
        
	}
	
	public void learn(){
		String[] command = new String[3];
		command[0]="python";
		command[1]=abspath + "/exrec.py";
		
		command[2] =abspath + "/Training/ExpressionDataSet";
		Process process=null;

		try {
			process = Runtime.getRuntime().exec(command);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			process.waitFor();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		process.destroy();

		IplImage grayImg=null;
		IplImage[] imgs = loadImage();
	 
		MatVector images = new MatVector(nFaces);
		int counter =0;
		for(IplImage img:imgs){
			grayImg = IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);
			cvCvtColor(img, grayImg, CV_BGR2GRAY);
			images.put(counter, grayImg);
			counter++;
		}


		CvMat x=grayImg.asCvMat();
		rows = x.rows();
		classifier.train(images, toIntArray(classOrder));
	}
	
	/*
	 * converts list to array
	 */
	public static int[] toIntArray(List<Integer> list){
		int[] ret = new int[list.size()];
		for(int i = 0;i < ret.length;i++)
			ret[i] = list.get(i);
		return ret;
	}
	
	/*
	 * save current model
	 */
	public void saveModel(){
		learn();	
		classifier.save("classifier_face_H_Sa.yml");	
	}
	
	public void loadModel(){
		classifier1.load("classifier_face_F|H_Sa|Su.yml");
		classifier2.load("classifier_face_F_H.yml");
		classifier3.load("classifier_face_Sa_Su.yml");
		classifier4.load("classifier_face_F|Su_H|Sa.yml");
		classifier5.load("classifier_face_F_Su.yml");
		classifier6.load("classifier_face_H_Sa.yml");
		classifier7.load("classifier_face_F|Sa_H|Su.yml");
		classifier8.load("classifier_face_F_Sa.yml");
		classifier9.load("classifier_face_H_Su.yml");
		classifier10.load("classifier_face_F_H_Sa_Su.yml");
		
	}
	
	/*
     * recognizes a given image(image path) based on existing model
     */
	private void recognize(FaceRecognizer a,String s,int c){
		IplImage img,grayImg;
		img = cvLoadImage(s);
		int len = s.length();
		String sub = s.substring(0,len-4);
		sub=sub+"_preProcess"+s.substring(len-4,len);
		cvSaveImage(sub,img);
		preProcess(sub,c);
		img = cvLoadImage(sub);
		Mat x=Highgui.imread(sub);
		grayImg = IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);
		cvCvtColor(img, grayImg, CV_BGR2GRAY);
		loadModel();
		double[] confidence ={0};
		int[] predictedLabel ={-1};
		a.predict(grayImg,predictedLabel,confidence);
		conf=confidence[0];
		pred = predictedLabel[0];
	

		
	}
	
	
	public int rec1(String s){
		recognize(classifier1,s,3);
		switch(pred){
		case 1:
			recognize(classifier2,s,3);
			if (pred ==1){
				return 0;
			}
			return 1;
		case 2:
			recognize(classifier3,s,3);
			if (pred == 1){
				return 2;
			}
			return 3;
		}
		return -1;
	}
	
	public int rec2(String s){
		recognize(classifier4,s,3);
		switch(pred){
		case 1:
			recognize(classifier5,s,3);
			if (pred ==1){
				return 0;
			}
			return 3;
		case 2:
			recognize(classifier6,s,3);
			if (pred == 1){
				return 1;
			}
			return 2;
		}
		return -1;
	}
	
	public int rec3(String s){
		recognize(classifier7,s,3);
		switch(pred){
		case 1:
			recognize(classifier8,s,3);
			if (pred ==1){
				return 0;
			}
			return 2;
		case 2:
			recognize(classifier9,s,3);
			if (pred == 1){
				return 1;
			}
			return 3;
		}
		return -1;
	}
	
	public int rec4(String s){
		recognize(classifier10,s,3);
		switch(pred){
		case 1:
			return 0;
		case 2:
			return 1;
		case 3:
			return 2;
		case 4:
			return 3;
		}
		return -1;
	}
	
	public String rec(String s){
		double[] class_dist = new double[4];
		double[] prob = new double[4];
		double[] count = new double[4];  // number of classifiers predicting a class
		double total_dist=0;
		String[] ex = {"fear","happy","sad","suprise"};
		for(int i=0;i<4;i++){
			class_dist[i] = 0;
			count[i]=0;
		}
		int x;
		double min_dist;
		x=rec1(s);
		class_dist[x]  += conf;
		total_dist += conf;
		count[x]+=1;
		System.out.println("Classifier 1 predicts " + ex[x] + " with a conf of " + conf);
		x=rec2(s);
		class_dist[x]  += conf;
		total_dist += conf;
		count[x]+=1;
		System.out.println("Classifier 2 predicts " + ex[x] + " with a conf of " + conf);
		x=rec3(s);
		class_dist[x]  += conf;
		total_dist += conf;
		count[x]+=1;
		System.out.println("Classifier 3 predicts " + ex[x]+ " with a conf of " + conf );
		x=rec4(s);
		class_dist[x]  += conf;
		total_dist += conf;
		count[x]+=1;
		System.out.println("Classifier 4 predicts " + ex[x] + " with a conf of " + conf);
		
		double[] p = new double[4];
		for (int i=0; i<4;i++){
			p[i] = 0.7*(count[i]/4)+0.3*(1-(class_dist[i]/total_dist)) ;
			
		}
		min_dist = Double.MAX_VALUE;
		int p_indx =0;
		for(int i=0;i<4;i++){
			if (p[i] > p[p_indx]){
			p_indx=i;				
			}
		}
		return ex[p_indx];
		
	}
	
	
	/*
	 * train/retrain model
	 */
	public void train(){
		saveModel();
	}

	public static void main(String arg[]){
		FaceEx x=new FaceEx();
		
		
       System.out.println("Most Likely Prediction is " +x.rec("Training/ExpressionDataSet/test1.jpg"));
	
		
	}

}

