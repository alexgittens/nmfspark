/* Computes a NMF factorization of a nearly-separable matrix, using the 
algorithm of 
``Scalable methods for nonnegative matrix factorizations of near-separable
tall-and-skinny matrices. Benson et al.''

*/

package org.apache.spark.mllib

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.linalg.{DenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow}
import edu.berkeley.cs.amplab.mlmatrix.{RowPartitionedMatrix, RowPartition, modifiedTSQR} 
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, diag}

import org.apache.spark.mllib.optimization.NNLS

import org.nersc.io._

import java.util.Calendar
import java.text.SimpleDateFormat

object nmf {

    def report(message: String, verbose: Boolean = true) = {
        val now = Calendar.getInstance().getTime()
        val formatter = new SimpleDateFormat("H:m:s")
        if (verbose) {
            println("STATUS REPORT (" + formatter.format(now) + "): " + message)
        }
    }

    case class NMFDecomposition(colbasis : DenseMatrix, loadings : DenseMatrix) {
        def H : DenseMatrix = colbasis
        def W : DenseMatrix = loadings
    }

    def fromBreeze(mat: BDM[Double]): DenseMatrix = {
        new DenseMatrix(mat.rows, mat.cols, mat.data, mat.isTranspose)
    }

    def main(args: Array[String]) = {
        val conf = new SparkConf().setAppName("nearSeparableNMF")
        val sc = new SparkContext(conf)
        sys.addShutdownHook( { sc.stop() } )
        appMain(sc, args)
    }

    def appMain(sc: SparkContext, args: Array[String]) = {
        val inpath = args(0)
        val variable = args(1)
        val numrows = args(2).toLong
        val numcols = args(3).toInt
        val partitions = args(4).toLong
        val rank = args(5).toInt
        val outdest = args(6)

        val temprows = read.h5read_irow(sc, inpath, variable, partitions)
        val A = loadH5Input(sc, inpath, variable, numrows, numcols, partitions)
        A.rdd.cache() // store deserialized in memory 
        A.rdd.count() // don't explicitly materialize, since we want to do pseudo one-pass, so let TSQR pay price of loading data

        // note this R is not reproducible between runs because it depends on the row partitioning
        // and the tree reduction order
        report("calling TSQR")
        val (colnorms, rmat) = new modifiedTSQR().qrR(A)
        report("TSQR worked")
        val normalizedrmat = rmat * diag( BDV.ones[Double](rmat.cols) :/ colnorms)
        report("starting xray")
        val extremalcolindices = xray.computeXray(normalizedrmat, rank)
        report("ending xray")

        report("Computing H from the small R matrix")
        val H = computeH(rmat, extremalcolindices)
        report("Done computing H")

/* THIS IS TOO LARGE (87GB, and too many indices) to collect for 1.6TB dayabay dataset 
        // We make a pass back over the data to extract the columns we care about
        // apparently RowPartitionedMatrix.collect is quite inefficient
        val W = A.mapPartitions( mat => { 
            val newMat = BDM.zeros[Double](mat.rows, extremalcolindices.length)
            (0 until extremalcolindices.length).foreach { 
                colidx => newMat(::, colidx) := mat(::, extremalcolindices(colidx)) }
            newMat
          }).collect()
          */

        // TODO: check correctness
        // one way to check rmat is correct is to right multiply A by its inverse and check for orthogonality
        // check the colnorms by explicitly constructing them

    }

    // return argmin || R - R[::, colindices]*H ||_F s.t. H >= 0
    def computeH(R : BDM[Double], colindices: Array[Int]) : BDM[Double] = {
    /*
        val RK = BDM.zeros[Double](R.rows, colindices.length)
        (0 until colindices.length).foreach { colidx => RK(::, colidx) := R(::, colindices(colidx)) }
        */
        val RK = BDM.horzcat( (0 until colindices.length).toArray.map( colidx => R(::, colindices(colidx)).asDenseMatrix.t ):_* )

        val H = BDM.zeros[Double](RK.cols, R.cols)
        val ws = NNLS.createWorkspace(RK.cols)
        val ata = (RK.t*RK).toArray

        for(colidx <- 0 until R.cols) {
            val atb = (RK.t*R(::, colidx)).toArray
            val h = NNLS.solve(ata, atb, ws)
            H(::, colidx) := BDV(h)
        }
        H
    }

    // note numrows, numcols are currently ignored, and the dimensions are taken from the variable itself
    def loadH5Input(sc: SparkContext, inpath: String, variable: String, numrows: Long, numcols: Int, repartition: Long) = {
        val temprows = read.h5read_irow(sc, inpath, variable, repartition)

        def buildRowBlock(iter: Iterator[IndexedRow]) : Iterator[RowPartition] = {
            val mat = BDM.zeros[Double](iter.length, numcols)
            for(rowidx <- 0 until iter.length) {
                val currow = iter.next.vector.toBreeze.asInstanceOf[BDV[Double]].t
                mat(rowidx, ::) := currow
            }
            Array(RowPartition(mat)).toIterator
        }

        new RowPartitionedMatrix(temprows.mapPartitions(buildRowBlock))
    }
}

