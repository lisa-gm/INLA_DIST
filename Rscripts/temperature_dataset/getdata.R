
library(data.table)
library(sp)

#(wdir <- here::here("rcode", "application"))
wdir <- getwd()

w2data.fl <- file.path(wdir, "w2data.rds")
wout.fl <- file.path(wdir, "wtavg.rds")
stations.fl <- file.path(wdir, "stations.rds")

### base URL
ghcnd <- "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"

##########################################################################
########### S T A T I O N S
##########################################################################

### file with inoformation on all the stations 
sfl <- "ghcnd-stations.txt"
loc.sfl <- file.path(wdir, sfl)
if(!file.exists(loc.sfl))
    utils::download.file(
               url = paste0(ghcnd, sfl),
               destfile = loc.sfl)

### width of the colums in the file:          
(ws <- diff(c(0,11,20,30,37,71,75,79,85)))

### read station information: longitute, latitude & altitude information
all.stations <- read.fwf(loc.sfl, ws[1:4])
colnames(all.stations) <- c('station', 'latitude', 'longitude', 'elevation')

### deal with sp and projection
coordinates(all.stations) <- ~ longitude + latitude
all.stations@proj4string <- CRS('+proj=longlat +datum=WGS84')

### index of stations with 'US' code
ii0us <- which(substr(all.stations$station, 1, 2)=='US')

## define a box around US main territory
box0ll <- SpatialPolygons(list(Polygons(list(Polygon(
    cbind(c(-130, -60, -60, -130, -130),
          c(50, 50, 23, 23, 50)))), '0')),
    proj4string=all.stations@proj4string)

ii1us <- which(!is.na(over(all.stations[ii0us,], box0ll)))

##################################################################
### Elevation grid (to fix unknown elevation for some stations)
##################################################################
options(timeout = max(1000, getOption("timeout")))

efl <- "ETOPO2.RData"
loc.efl <- file.path(wdir, efl)
if(!file.exists(loc.efl))
    utils::download.file(
               url = paste0("http://leesj.sites.oasis.unc.edu/",
                            "FETCH/GRAB/RPACKAGES/", efl), 
               destfile = loc.efl)

### load the ETOPO  data
load(loc.efl)
cat("ETOPO data dimention:", dim(ETOPO2), "\n")

### extract the longitude (and fix) and latitude
elon <- attr(ETOPO2, "lon")
elon[elon>=180] <- 180-rev(elon[elon>=180]) 
elat <- attr(ETOPO2, "lat")

### fix the order of the lines
ETOPO2 <- ETOPO2[, ncol(ETOPO2):1]

alocs.ll <- sp::coordinates(all.stations)

ij <- list(i=findInterval(alocs.ll[,1], c(-180, elon+1/60)),
           j=findInterval(alocs.ll[,2], elat))

etopoll <- sapply(1:nrow(alocs.ll), function(i) ETOPO2[ij$i[i], ij$j[i]])

### index of the elevation data to fix
ii.to.fix <- which(all.stations$elevation<(-999))

### fix the elevation data 
all.stations$elevation[ii.to.fix] <- etopoll[ii.to.fix]

##########################################################
### daily weather data for a given year
##########################################################

dfl <- "2022.csv.gz"
loc.dfl <- file.path(wdir, dfl)
if(!file.exists(loc.dfl))
    utils::download.file(
               url = paste0(ghcnd, "by_year/", dfl), 
               destfile = loc.dfl)

if(file.exists(w2data.fl)) {

    w2d <- readRDS(w2data.fl)
    
} else {

    long2wide <- function(fl) {        
        ldata0 <- data.table::fread(fl, data.table = FALSE)
        cat("Read", nrow(ldata0), "lines\n")
        i0 <- (ldata0$V6=='') &
            (ldata0$V1 %in% all.stations$station[ii0us[ii1us]])
        ii <- which(i0 & (ldata0$V3 %in% c("TMIN", "TMAX")))
        cat("Will select", length(ii), "observations\n")
        t0 <- Sys.time()
        w <- tapply(ldata0[ii, 4],
                    ldata0[ii, c(1,2,3)],
                    as.integer)
        t1 <- Sys.time()
        cat("Dim =", dim(w), "")
        print(t1 - t0)
        return(w)
    }

    w2d <- long2wide(loc.dfl)
    
    saveRDS(object = w2d,
            file = w2data.fl)

}

wtavg0 <- (w2d[,,1] + w2d[,,2])/20
dim(wtavg0)

nd.s <- rowSums(!is.na(wtavg0))
summary(nd.s)

table(cut(nd.s, c(0, 14, 61, 182, 300, 350, 365), include.lowest = TRUE))
plot(table(cut(nd.s, 5*(0:73), include.lowest = TRUE)))

### Select data on stations with more than 182 non-missing tavg
str(ii.ds <- which(nd.s>182))

wtavg <- wtavg0[ii.ds, ]
saveRDS(object = wtavg,
        file = wout.fl)

### stations coordinates in Mollweide projection with units in km
ii.ss.d <- pmatch(dimnames(wtavg)[[1]], 
                  all.stations$station[ii0us[ii1us]])
str(ii.ss.d)
summary(ii.ss.d)

stations.mkm <- spTransform(
    all.stations[ii0us[ii1us][ii.ss.d], ], "+proj=moll +units=km")

if(FALSE)
    plot(stations.mkm)

saveRDS(object = stations.mkm,
        file = stations.fl)

