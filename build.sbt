version := "0.0.1"
scalaVersion := "2.10.4"
libraryDependencies += "org.apache.spark" %% "spark-core" % "1.5.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.1" % "provided"
libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"
libraryDependencies += "org.msgpack" %% "msgpack-scala" % "0.6.11"

val awskey = System.getenv("AWS_ACCESS_KEY_ID")
val awssecretkey = System.getenv("AWS_SECRET_ACCESS_KEY")

lazy val runTest = taskKey[Unit]("test NMF")
runTest <<= (assembly in Compile) map {
  (jarFile: File) => s"src/testNMF.sh ${jarFile}" !
}

lazy val submit = taskKey[Unit]("compute NMF on large dataset")
submit <<= (assembly in Compile) map {
  (jarFile: File) => s"src/computeNMF.sh ${jarFile}" !
} 
