FROM node:8-alpine

RUN mkdir /app
WORKDIR /app
COPY /src /app/src
COPY ["package.json", "package-lock.json*", "./"]

RUN npm install --production --silent && mv node_modules ../

# Expose PORT 3000 on our virtual machine so we can run our server
EXPOSE 3000

ENTRYPOINT ["npm", "start"]
