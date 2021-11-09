/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.dataloader;

import com.google.gson.Gson;
import com.mycompany.neuralnetwork.matrix.Matrix;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Arrays;
import java.util.stream.Collectors;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author dmytr
 */
public class DataLoader {
    
    public static List<INDArray[]> loadTrainingDataAsINDArrays(int packages) throws IOException{
        String fileName="src\\main\\resources\\data\\mnist_training_data_";
        List<INDArray[]> result=loadDataAsINDArrays(fileName+"1.json");
        for(int i=1;i<packages;i++){
            try{
                result.addAll(loadDataAsINDArrays(fileName+(i+1)+".json"));
            } catch(FileNotFoundException e){
                System.out.println(e.getMessage());
                break;
            }
        }
        return result;
    }
    
    public static List<INDArray[]> loadTestDataAsINDArrays() throws IOException{
        return loadDataAsINDArrays("src\\main\\resources\\data\\mnist_test_data.json");
    }
    
    public static List<INDArray[]> loadValidationDataAsINDArrays() throws IOException{
        return loadDataAsINDArrays("src\\main\\resources\\data\\mnist_validation_data.json");
    }
    
    public static List<Matrix[]> loadTrainingDataAsMatrices(int packages) throws IOException{
        String fileName="src\\main\\resources\\data\\mnist_training_data_";
        List<Matrix[]> result=loadDataAsMatrices(fileName+"1.json");
        for(int i=1;i<packages;i++){
            try{
                result.addAll(loadDataAsMatrices(fileName+(i+1)+".json"));
            } catch(FileNotFoundException e){
                System.out.println(e.getMessage());
                break;
            }
        }
        return result;
    }
    
    public static List<Matrix[]> loadTestDataAsMatrices() throws IOException{
        return loadDataAsMatrices("src\\main\\resources\\data\\mnist_test_data.json");
    }
    
    public static List<Matrix[]> loadValidationData() throws IOException{
        return loadDataAsMatrices("src\\main\\resources\\data\\mnist_validation_data.json");
    }
    
    public static List<Matrix[]> loadDataAsMatrices(String fileName) throws IOException{
        File file=new File(fileName);
        System.out.println(file.getAbsolutePath());
        FileReader fr=new FileReader(fileName);
        BufferedReader br=new BufferedReader(fr);
        String s=null;
        StringBuilder text=new StringBuilder();
        long beginPoint=System.currentTimeMillis();
        while((s=br.readLine())!=null){
            text.append(s);
        }
        br.close();
        fr.close();
        String[] matrixArr=text.substring(2, text.length()-2).toString().split("\\], \\[");
        long readingEnded=System.currentTimeMillis()-beginPoint;
        System.out.println("Reading ended:"+readingEnded);        
        Gson gson=new Gson();
        List<Matrix[]> result=(List<Matrix[]>)Arrays.stream(matrixArr).map(str->
                gson.fromJson(new StringBuilder(str).insert(0,"[").append("]").toString(),
                        Matrix[].class)).map(ms->{
                            ms[0].setElType(Double.class);
                            ms[0].setElements(ms[0].getElements().stream().map(e->e.doubleValue()).collect(Collectors.toList()));
                            ms[1].setElType(Double.class);
                            ms[1].setElements(ms[1].getElements().stream().map(e->e.doubleValue()).collect(Collectors.toList()));
                            return ms;
                                }).collect(Collectors.toList());
        System.out.println("Transforming to list ended: "+((System.currentTimeMillis()-readingEnded)/1000));
        return result;
    }
    
    public static List<INDArray[]> loadDataAsINDArrays(String fileName) throws IOException{        
        return loadDataAsMatrices(fileName).stream().map(m->
                new INDArray[]{Nd4j.create(m[0].getElements().stream().mapToDouble(e->e.doubleValue()).toArray(),
                        Arrays.stream(m[0].getShape()).mapToLong(i->(long)i).toArray(),DataType.DOUBLE),
                    Nd4j.create(m[1].getElements().stream().mapToDouble(e->e.doubleValue()).toArray(),
                        Arrays.stream(m[1].getShape()).mapToLong(i->(long)i).toArray(),DataType.DOUBLE)})
                .collect(Collectors.toList());
    }
}
