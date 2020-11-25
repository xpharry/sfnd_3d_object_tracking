#ifndef CSVWRITER_H
#define CSVWRITER_H

// A class to create and write data in a csv file.
class CSVWriter
{
	std::string fileName;
	std::string delimeter;
	int linesCount;

public:
	CSVWriter(std::string filename, std::string delm = ";") :
		fileName(filename), delimeter(delm), linesCount(0)
	{}
	/*
	 * Member function to store a range as comma seperated value
	 */
	template<typename T>
	void addDatainRow(T first, T last);
};


/*
* This Function accepts a range and appends all the elements in the range
* to the last row, seperated by delimeter (Default is comma)
*/
template<typename T>
void CSVWriter::addDatainRow(T first, T last)
{
    std::fstream file;
    // Open the file in truncate mode if first line else in Append Mode
    file.open(fileName, std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));
    // Iterate over the range and add each lement to file seperated by delimeter.
    for (; first != last; )
    {
        file << *first;
        if (++first != last)
            file << delimeter;
    }
    file << "\n";
    linesCount++;
    // Close the file
    file.close();
}

#endif /* CSVWRITER_H */
