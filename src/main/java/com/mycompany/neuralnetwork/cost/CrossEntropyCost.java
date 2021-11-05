/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.cost;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.ops.NDMath;

/**
 *
 * @author dmytr
 */
public class CrossEntropyCost implements Cost{
    
    public double fun(INDArray actual_output, INDArray expected_output){
        NDMath ndmath=new NDMath();
        return expected_output.mul(-1).mul(ndmath.log(actual_output)).sub(
                expected_output.mul(-1).add(1).mul(ndmath.log(actual_output.mul(-1).add(1)))).sum().data().asDouble()[0];
    }
    
    public INDArray delta(INDArray actual_output, INDArray expected_output, INDArray z){
        return actual_output.sub(expected_output);
    }
}
