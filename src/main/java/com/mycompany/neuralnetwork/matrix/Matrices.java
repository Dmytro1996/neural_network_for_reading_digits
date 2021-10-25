/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.matrix;

import com.mycompany.neuralnetwork.exceptions.MatrixException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.UnaryOperator;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 *
 * @author dmytr
 */
public class Matrices {
    
    public static <T extends Number> Number turnToElType(Class<T> elType, double d){
        switch(elType.getSimpleName()){
            case "Integer":
                    return (int)d;
                case "Long":
                    return (long)d;
                case "Float":
                    return (float)d;
                default:
                    return d;
        }
    }
    
    public static <T extends Number> List<Number> fillMatrix(Class<T> type,int size) throws MatrixException{
        Random rand=new Random();
        switch(type.getSimpleName()){
                case "Integer":
                    return rand.ints().limit(size).boxed().collect(Collectors.toList());
                case "Double":
                    return Stream.generate(()->rand.nextGaussian()).limit(size).collect(Collectors.toList());
                case "Float":
                    return Stream.generate(()->rand.nextGaussian()).limit(size).map(f->f.floatValue()).collect(Collectors.toList());
                case "Long":
                    return rand.longs().limit(size).boxed().collect(Collectors.toList());
                default:
                    throw new MatrixException("Wrong data type");
       }
    }
    
    public static void checkMatrixesForAddition(Matrix firstMatrix,Matrix secondMatrix) throws MatrixException{
        if(firstMatrix.getSize()!=secondMatrix.getSize() 
                && firstMatrix.getAxes()!=secondMatrix.getAxes() 
                && Arrays.equals(firstMatrix.getShape(), secondMatrix.getShape())){
            throw new MatrixException("Matrixes are of different sizes");
        }
    }
    
    public static void checkMatrixesForMultiplication(Matrix firstMatrix,Matrix secondMatrix) throws MatrixException{
        for(int i=0;i<Math.min(firstMatrix.getAxes(), secondMatrix.getAxes());i++){
            if(firstMatrix.getShape()[i]!=secondMatrix.getShape()[i]){
                throw new MatrixException("Matrixes are of different shapes");
            }
        }
    }
    
    public static Matrix copyOf(Matrix matrix) throws MatrixException{
        return new Matrix(matrix.getElements(),matrix.getShape(),matrix.getElType());
    }
    
    public static Matrix matrixMult(Matrix...matrices) throws MatrixException{
        return Arrays.stream(matrices).skip(1).reduce(Matrices.copyOf(matrices[0]), (acc,x)->{try {
            acc.matrixMult(x);
            } catch (MatrixException ex) {
                Logger.getLogger(Matrices.class.getName()).log(Level.SEVERE, null, ex);
            }
            return acc;});
    }
    
    public static Matrix add(Matrix...matrixes) throws MatrixException{//think about replacing stream with for loop
        //long beginPoint=System.currentTimeMillis();
        if(Arrays.stream(matrixes).skip(1).anyMatch(m->
                !Arrays.equals(m.getShape(),matrixes[0].getShape()))){
            throw new MatrixException("Matrixes are of different shapes");
        }
        Matrix result=Arrays.stream(matrixes).skip(1).reduce(Matrices.copyOf(matrixes[0]), (acc,x)->{try {
            acc.add(x);
            } catch (MatrixException ex) {
                Logger.getLogger(Matrices.class.getName()).log(Level.SEVERE, null, ex);
            }
            return acc;});
        //System.out.println("Addition"+(System.currentTimeMillis()-beginPoint));
        return result;
    }
    
    public static Matrix subtract(Matrix...matrixes) throws MatrixException{
        if(Arrays.stream(matrixes).skip(1).anyMatch(m->
                !Arrays.equals(m.getShape(),matrixes[0].getShape()))){
            throw new MatrixException("Matrixes are of different shapes");
        }
        return Arrays.stream(matrixes).skip(1).reduce(Matrices.copyOf(matrixes[0]), (acc,x)->{try {
            acc.subtract(x);
            } catch (MatrixException ex) {
                Logger.getLogger(Matrices.class.getName()).log(Level.SEVERE, null, ex);
            }
            return acc;});
    }
    
    public static Matrix multiply(Matrix...matrixes) throws MatrixException{
        //long beginPoint=System.currentTimeMillis();
        if(Arrays.stream(matrixes).skip(1).anyMatch(m->
                m.getShape()[0]!=matrixes[0].getShape()[0])){
            throw new MatrixException("Matrixes are of incompatible shapes");
        }
        Matrix result=Arrays.stream(matrixes).skip(1).reduce(Matrices.copyOf(matrixes[0]), (acc,x)->{try {
            acc.multiply(x);
            } catch (MatrixException ex) {
                Logger.getLogger(Matrices.class.getName()).log(Level.SEVERE, null, ex);
            }
            return acc;});
        //System.out.println("Multiplication"+(System.currentTimeMillis()-beginPoint));
        return result;
    }
    
    public static Matrix multiply(Matrix matrix, double...nums) throws MatrixException{
        Matrix result=Matrices.copyOf(matrix);
        for(double d:nums){
            result.multiply(d);
        }
        return result;
    }
    
    public static Matrix divide(Matrix...matrixes) throws MatrixException{
        if(Arrays.stream(matrixes).skip(1).anyMatch(m->
                m.getShape()[0]!=matrixes[0].getShape()[0])){
            throw new MatrixException("Matrixes are of incompatible shapes");
        }
        return Arrays.stream(matrixes).skip(1).reduce(Matrices.copyOf(matrixes[0]), (acc,x)->{try {
            acc.divide(x);
            } catch (MatrixException ex) {
                Logger.getLogger(Matrices.class.getName()).log(Level.SEVERE, null, ex);
            }
            return acc;});
    }
    
    public static Matrix applyFunc(Matrix matrix,UnaryOperator<Double> func) throws MatrixException{
        Matrix result=copyOf(matrix);
        result.applyFunc(func);
        return result;
    }
    
    public static Matrix transpose(Matrix matrix) throws MatrixException{
        //long beginPoint=System.currentTimeMillis();
        Matrix result=copyOf(matrix);
        result.transpose();
        //System.out.println("Transpose"+(System.currentTimeMillis()-beginPoint));
        return result;
    }
    
    public static String matrixToString(Matrix matrix){
        StringBuilder result=new StringBuilder();
        int[] closingPoints=new int[matrix.getAxes()];
        for(int i=matrix.getAxes()-1;i>=0;i--){
            result.append("[");
            int newClosingPoint=1;
            for(int j=closingPoints.length-1;j>=0;j--){
                if(closingPoints[j]==0)break;
                newClosingPoint*=closingPoints[j];
            }
            closingPoints[i]=newClosingPoint*matrix.getShape()[i];
        }
        for(int i=0;i<matrix.getSize();i++){
            result.append(matrix.getElements().get(i));
            int bracketsClosed=0;
            for(int j=closingPoints.length-1;j>=0;j--){
                if((i+1)>=closingPoints[j] && (i+1)%closingPoints[j]==0){
                    result.append("]");
                    if(j==closingPoints.length-1  && i!=(matrix.getSize()-1))result.append(",\n");
                    bracketsClosed++;
                }
            }
            if(i!=(matrix.getSize()-1)){
                if(result.charAt(result.length()-1)!='\n')result.append(",");
                for(int j=0;j<bracketsClosed;j++){
                    result.append("[");
                }
            }
        }
        return result.toString();
    }   
    
    public static List<Number> combineElements(List<List<Number>> matrixElems) throws Exception{
        List<Number> result=new ArrayList<>();
        if(matrixElems.stream().anyMatch(m->m.size()!=matrixElems.get(0).size())){
            throw new Exception("Lists of elements are of different sizes");
        }
        result=matrixElems.stream().reduce(new ArrayList<Number>(),(acc,x)->{
            for(int i=0;i<x.size();i++){
                if(acc.size()<=i)acc.add(0);
                acc.set(i,acc.get(i).doubleValue()+x.get(i).doubleValue());
            }
            return acc;
        });
        return result;
    }
    
}
