//
//  Float16.swift
//  MetalMatrixMult
//
//  Created by Anton Akusok on 06/07/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import Foundation
import Accelerate

public typealias Float16 = UInt16

/*
 Uses vImage to convert a buffer of float16 values to regular Swift Floats.
 */
public func float16to32(_ input: UnsafeMutableRawPointer, count: Int) -> UnsafeMutableBufferPointer<Float> {
    let p = UnsafeMutablePointer<Float>.allocate(capacity: count)
    let output = UnsafeMutableBufferPointer(start: p, count: count)
    
    var bufferFloat16 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 2)
    var bufferFloat32 = vImage_Buffer(data: p, height: 1, width: UInt(count), rowBytes: count * 4)
    
    if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
        print("Error converting float16 to float32")
    }
    return output
}

/*
 Uses vImage to convert an array of Swift floats into a buffer of float16s.
 */
public func float32to16(_ input: UnsafeMutablePointer<Float>, count: Int) -> UnsafeMutableBufferPointer<Float16> {
    let p = UnsafeMutablePointer<Float16>.allocate(capacity: count)
    let output = UnsafeMutableBufferPointer(start: p, count: count)
    
    var bufferFloat32 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 4)
    var bufferFloat16 = vImage_Buffer(data: p, height: 1, width: UInt(count), rowBytes: count * 2)
    
    if vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0) != kvImageNoError {
        print("Error converting float32 to float16")
    }
    return output
}
