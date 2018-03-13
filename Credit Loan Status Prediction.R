#Predict Loan Status

library(data.table)
train <- fread(file.choose(), stringsAsFactors = T)
test <- fread(file.choose(), stringsAsFactors = T)

#Deleting Empty rows
train <- head(train,100000)

#Deleting duplicated rows
train <- subset(train, !duplicated(`Loan ID`))

#Formatting Years in current job column
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "8 years"] <- "8"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "10+ years"] <- "8"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "3 years"] <- "3"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "5 years"] <- "5"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "6 years"] <- "6"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "7 years"] <- "7"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "9 years"] <- "9"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "< 1 year"] <- "0.5"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "1 year"] <- "1"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "2 years"] <- "2"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "4 years"] <- "4"
levels(train$`Years in current job`)[levels(train$`Years in current job`) ==  "n/a"] <- "0"

#Encoding categorical variables
train$`Loan Status` <- factor(train$`Loan Status`, labels = c(1,2), levels = c('Fully Paid','Charged Off'))
train$Term <- factor(train$Term, labels = c(1,2), levels = c('Short Term','Long Term'))
train$`Home Ownership` <- factor(train$`Home Ownership`, labels = c(1,2,3,4), levels = c('Home Mortgage','Own Home','Rent','HaveMortgage') )
train$Purpose <- factor(train$Purpose, labels = c(1:17), c('Home Improvements','Debt Consolidation','Buy House','Business Loan','Buy a Car','Educational Expenses','Medical Bills','Take a Trip','major_purchase','moving','vacation','renewable_energy','small_business','wedding','','Other','other'))
#Since 15,16,17 are in to same category
levels(train$Purpose)[levels(train$Purpose) ==  "15"] <- "17"
levels(train$Purpose)[levels(train$Purpose) ==  "16"] <- "17"

#Converting factore into integers
train$`Loan Status` <- as.integer(train$`Loan Status`)
train$Term <- as.integer(train$Term)
train$`Home Ownership` <- as.integer(train$`Home Ownership`)
train$Purpose <- as.integer(train$Purpose)
train$`Years in current job` <- as.integer(train$`Years in current job`)


#Finding Missing Values
sort(sapply(train, function(x) { sum(is.na(x)) }), decreasing=TRUE)

#Note: Credit Score and Annual Income has same number of missing values

############################# Don't run this its taking tooooo long #########################################
#Imputing Missing Values
library(mice)
imputed_data <- mice(train[,c(6,7,17,18,19)], m=5, maxit = 50, method = 'rf', seed = 500)
train[,c(6,7,17,18,19)] <- complete(imputed_data, 2)
############################## Taking toooo long ############################################################

#imputing Na values using Amelia
install.packages("Amelia")
library(Amelia)
amelia_fit <- amelia(train[,c(6,7,17,18,19)], m=5, parallel = "multicore")
train[,c(6,7,17,18,19)] <- amelia_fit$imputations[[2]]
########################## Unable to insert values in bancruptcies :-( #######################################

#Finding mode for the bancruptcies
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
mode(train$Bankruptcies)

#Inserting missing values with mode in bancrupcies (here mode is 0)
train$Bankruptcies[is.na(train$Bankruptcies)] <- 0
