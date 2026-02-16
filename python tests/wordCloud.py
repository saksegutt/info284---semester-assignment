# Source - https://stackoverflow.com/a/46203314
# Posted by Anil_M, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-16, License - CC BY-SA 3.0

import csv
from wordcloud import WordCloud


#read first column of csv file to string of words seperated
#by tab
your_list = []
with open('reviews.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    your_list = '\t'.join([i[0] for i in reader])


# Generate a word cloud image
wordcloud = WordCloud().generate(your_list)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(your_list)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()
