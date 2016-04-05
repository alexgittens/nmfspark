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
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, qr, svd}

import java.util.Calendar
import java.text.SimpleDateFormat

import org.nersc.io._

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
        val partitions = args(4).toInt
        val rank = args(5).toInt
        val outdest = args(6)

        Ral mat = loadH5Input(sc, inpath, variable, numrows, numcols, partitions)

        // note this R is not reproducible between runs because it depends on the row partitioning
        // and the tree reduction order
        val (colnorms, rmat) = new modifiedTSQR().qrR(mat)

        // one way to check rmat is correct is to right multiply by its inverse and check for orthogonality
        // check the colnorms by explicitly constructing them

        val extremalcols = xray(normalizedrmat)
    }

    // note numrows, numcols are currently ignored, and the dimensions are taken from the variable itself
    def loadH5Input(sc: SparkContext, inpath: String, variable: String, numrows: Long, numcols: Int, repartition: Long) = {
        val temprows = read.h5read_imat(sc, inpath, variable, repartition).rows

        def buildRowBlock(iter: Iterator[IndexedRow]) : Iterator[RowPartition] = {
            val numrows = iter.length
            val firstrow = iter.next.vector.toBreeze.asInstanceOf[BDV[Double]]
            val numcols = firstrow.length

            val tempblock = BDM.zeros[Double](numrows, numcols)
            tempblock(0, ::) := firstrow.t
            for (rowidx <- 1 until iter.length) {
                tempblock(rowidx, ::) := iter.next.vector.toBreeze.asInstanceOf[BDV[Double]].t
            }

            Array(RowPartition(tempblock)).toIterator
        }

        new RowPartitionedMatrix(temprows.mapPartitions(buildRowBlock))
    }
}

