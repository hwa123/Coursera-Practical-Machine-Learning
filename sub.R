answers <- predictRf_testing

pml_write_files = function(x){
  n = length(x)
  
  if (!file.exists("./results")){
    dir.create("./results")
  }
  
  setwd("~/R/Coursera/Practical Machine Learning/results")
  
    for(i in 1:n){
      filename = paste0("problem_id_",i,".txt")
      write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
  }

pml_write_files(answers)