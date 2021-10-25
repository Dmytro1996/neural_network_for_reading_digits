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
import java.util.LinkedList;
import java.util.List;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;

/**
 *
 * @author dmytr
 */
public class DataLoader {
    
    public static List<Matrix[]> loadTrainingData(int packages) throws IOException{
        String fileName="src\\main\\resources\\data\\mnist_training_data_";
        List<Matrix[]> result=loadData(fileName+"1.json");
        for(int i=1;i<packages;i++){
            result.addAll(loadData(fileName+(i+1)+".json"));
        }
        return result;
    }
    
    public static List<Matrix[]> loadTestData() throws IOException{
        return loadData("src\\main\\resources\\data\\mnist_test_data.json");
    }
    
    public static List<Matrix[]> loadValidationData() throws IOException{
        return loadData("src\\main\\resources\\data\\mnist_validation_data_by_line.json");
        //return loadData("C:\\Users\\dmytr\\OneDrive\\Documents\\Python\\train_data_short_json.json");
    }
    
    public static List<Matrix[]> loadData(String fileName) throws IOException{
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
        List<Matrix[]> result=(List<Matrix[]>)Arrays.stream(matrixArr).limit(5000).map(str->
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
}
